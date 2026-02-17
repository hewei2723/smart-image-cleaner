import os
import sys
import json
import threading
import multiprocessing
import concurrent.futures
from collections import defaultdict
from PIL import Image
import imagehash
from flask import Flask, render_template, request, jsonify, send_file
import webbrowser
import cv2
import numpy as np

app = Flask(__name__)

# Global state
class ScanState:
    def __init__(self):
        self.scanning = False
        self.progress_msg = "等待开始"
        self.results = []
        self.folder_path = ""
        self.stop_flag = False
        self.mode = 'duplicate' # 'duplicate' or 'blur'

state = ScanState()

# --- Core Logic ---

def calculate_blur_score(path):
    try:
        # Read image using OpenCV
        # Handle unicode paths
        stream = open(path, "rb")
        bytes = bytearray(stream.read())
        numpyarray = np.asarray(bytes, dtype=np.uint8)
        img = cv2.imdecode(numpyarray, cv2.IMREAD_GRAYSCALE)
        stream.close()
        
        if img is None:
            return 0
            
        # Variance of Laplacian
        score = cv2.Laplacian(img, cv2.CV_64F).var()
        return score
    except Exception:
        return 0

def process_one_image(path, mode='duplicate'):
    try:
        size = os.path.getsize(path)
        mtime = os.path.getmtime(path)
        
        result = {
            'path': path,
            'size': size,
            'mtime': mtime,
            'error': None
        }

        if mode == 'duplicate':
            with Image.open(path) as img:
                width, height = img.size
                phash = imagehash.phash(img)
                result.update({
                    'width': width,
                    'height': height,
                    'phash': str(phash)
                })
        elif mode == 'blur':
            with Image.open(path) as img:
                width, height = img.size
                result.update({
                    'width': width,
                    'height': height
                })
            # Calculate blur separately
            score = calculate_blur_score(path)
            result['blur_score'] = score
            
        return result
            
    except Exception as e:
        return {
            'path': path,
            'error': str(e)
        }

class ImageInfo:
    def __init__(self, data):
        self.path = data['path']
        self.size = data.get('size', 0)
        self.mtime = data.get('mtime', 0)
        self.width = data.get('width', 0)
        self.height = data.get('height', 0)
        self.phash = None
        self.blur_score = data.get('blur_score', 0)
        
        phash_str = data.get('phash')
        if phash_str:
            try:
                self.phash = imagehash.hex_to_hash(phash_str)
            except:
                pass

    def to_dict(self):
        return {
            'path': self.path,
            'name': os.path.basename(self.path),
            'size': self.size,
            'mtime': self.mtime,
            'width': self.width,
            'height': self.height,
            'phash': str(self.phash) if self.phash else None,
            'blur_score': self.blur_score
        }

def scan_task(folder_path, mode='duplicate'):
    state.scanning = True
    state.folder_path = folder_path
    state.results = []
    state.mode = mode
    state.progress_msg = "正在初始化..."
    
    try:
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp'}
        cache_file = os.path.join(folder_path, ".dedup_cache.json")
        cache = {}
        
        # Load Cache
        try:
            if os.path.exists(cache_file):
                with open(cache_file, 'r', encoding='utf-8') as f:
                    cache = json.load(f)
        except Exception as e:
            print(f"Cache load error: {e}")

        # Step 1: Collect files
        files = []
        for root, _, filenames in os.walk(folder_path):
            for filename in filenames:
                if filename.startswith('.'): continue
                if os.path.splitext(filename)[1].lower() in valid_extensions:
                    files.append(os.path.join(root, filename))
        
        total = len(files)
        hashes = defaultdict(list)
        blur_list = []
        
        to_process = []
        cached_results = []
        
        state.progress_msg = f"正在检查缓存 ({total} 张图片)..."
        
        for file_path in files:
            if state.stop_flag: break
            try:
                stat = os.stat(file_path)
                size = stat.st_size
                mtime = stat.st_mtime
                
                if file_path in cache:
                    c_data = cache[file_path]
                    if c_data.get('size') == size and abs(c_data.get('mtime', 0) - mtime) < 0.1:
                        # Check if cache has required data for current mode
                        if mode == 'duplicate' and 'phash' in c_data:
                            cached_results.append(c_data)
                            continue
                        elif mode == 'blur' and 'blur_score' in c_data:
                            cached_results.append(c_data)
                            continue
            except:
                pass
            to_process.append(file_path)

        if state.stop_flag:
            state.scanning = False
            state.progress_msg = "已停止"
            return

        # Step 3: Compute
        if to_process:
            state.progress_msg = f"需要处理 {len(to_process)} 张新图片 (缓存命中 {len(cached_results)} 张)..."
            
            max_workers = max(1, multiprocessing.cpu_count() - 1)
            processed_count = 0
            
            with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
                # Need to pass mode to worker? process_one_image takes mode
                # Use lambda or functools.partial? No, just helper
                future_to_file = {executor.submit(process_one_image, f, mode): f for f in to_process}
                
                for future in concurrent.futures.as_completed(future_to_file):
                    if state.stop_flag: break
                    processed_count += 1
                    try:
                        result = future.result()
                        if not result.get('error'):
                            cached_results.append(result)
                            # Merge into cache (preserve existing keys if possible)
                            if result['path'] in cache:
                                cache[result['path']].update(result)
                            else:
                                cache[result['path']] = result
                    except Exception as exc:
                        print(f"Exception: {exc}")
                    
                    if processed_count % 10 == 0:
                        state.progress_msg = f"正在处理: {processed_count}/{len(to_process)}"
        
        if state.stop_flag:
            state.scanning = False
            state.progress_msg = "已停止"
            return

        # Step 4: Save Cache
        try:
            current_files_set = set(files)
            keys_to_remove = [k for k in cache.keys() if k not in current_files_set]
            for k in keys_to_remove:
                del cache[k]
                
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Cache save error: {e}")

        # Step 5: Process Results based on Mode
        state.progress_msg = "正在整理结果..."
        
        final_results = []
        
        if mode == 'duplicate':
            for data in cached_results:
                img_info = ImageInfo(data)
                if img_info.phash:
                    hashes[str(img_info.phash)].append(img_info)

            for hash_str, img_list in hashes.items():
                if len(img_list) > 1:
                    group_data = [img.to_dict() for img in img_list]
                    group_data.sort(key=lambda x: (x['width'] * x['height'], x['size']), reverse=True)
                    final_results.append({
                        'hash': hash_str,
                        'images': group_data,
                        'type': 'group'
                    })
            state.progress_msg = f"扫描完成，发现 {len(final_results)} 组重复图片"
            
        elif mode == 'blur':
            for data in cached_results:
                img_info = ImageInfo(data)
                if img_info.blur_score < 100: # Threshold for blur. <100 is usually blurry
                    final_results.append(img_info.to_dict())
            
            # Sort by blur score (ascending = most blurry first)
            final_results.sort(key=lambda x: x['blur_score'])
            state.progress_msg = f"扫描完成，发现 {len(final_results)} 张模糊图片"

        state.results = final_results
        
    except Exception as e:
        state.progress_msg = f"错误: {str(e)}"
    finally:
        state.scanning = False

# --- Flask Routes ---

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/scan', methods=['POST'])
def start_scan():
    data = request.json
    folder_path = data.get('path')
    mode = data.get('mode', 'duplicate') # Default to duplicate
    
    if not folder_path or not os.path.exists(folder_path):
        return jsonify({'error': '无效的文件夹路径'}), 400
    
    if state.scanning:
        return jsonify({'error': '扫描正在进行中'}), 400
    
    state.stop_flag = False
    thread = threading.Thread(target=scan_task, args=(folder_path, mode))
    thread.daemon = True
    thread.start()
    
    return jsonify({'status': 'started'})

@app.route('/api/status')
def get_status():
    return jsonify({
        'scanning': state.scanning,
        'message': state.progress_msg,
        'count': len(state.results),
        'mode': state.mode
    })

@app.route('/api/results')
def get_results():
    return jsonify({
        'mode': state.mode,
        'data': state.results
    })


@app.route('/api/image')
def get_image():
    path = request.args.get('path')
    if not path or not os.path.exists(path):
        return "File not found", 404
    return send_file(path)

@app.route('/api/delete', methods=['POST'])
def delete_images():
    data = request.json
    paths = data.get('paths', [])
    deleted = []
    errors = []
    
    for path in paths:
        try:
            if os.path.exists(path):
                os.remove(path) # Permanently delete. Or send to trash?
                # User asked for delete. Let's do permanent delete for now as standard python behavior, 
                # but maybe send2trash is safer? 
                # For this task, os.remove is standard.
                deleted.append(path)
        except Exception as e:
            errors.append({'path': path, 'error': str(e)})
            
    return jsonify({'deleted': deleted, 'errors': errors})

@app.route('/api/open_folder', methods=['POST'])
def open_folder():
    # Use tkinter to ask for directory
    import tkinter as tk
    from tkinter import filedialog
    
    # Create a hidden root window
    root = tk.Tk()
    root.withdraw()
    
    # Bring to front (hacky on Windows)
    root.attributes('-topmost', True)
    
    folder_path = filedialog.askdirectory()
    
    root.destroy()
    return jsonify({'path': folder_path})

if __name__ == '__main__':
    # Open browser automatically
    webbrowser.open('http://127.0.0.1:5000')
    app.run(debug=True, use_reloader=False)
