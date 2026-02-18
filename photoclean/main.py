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
import shutil

# Lazy import for heavy ML libraries
sentence_transformers = None
util = None

def get_model():
    global sentence_transformers, util
    if sentence_transformers is None:
        try:
            # Set model download path to local .models folder
            os.environ['SENTENCE_TRANSFORMERS_HOME'] = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.models')
            
            from sentence_transformers import SentenceTransformer, util as st_util
            sentence_transformers = SentenceTransformer
            util = st_util
        except ImportError:
            return None
    return sentence_transformers('clip-ViT-B-32')

# Configure Flask to look for templates in the same directory as this file
template_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates')
print(f"DEBUG: template_dir={template_dir}")
if os.path.exists(template_dir):
    print(f"DEBUG: template_dir exists")
    print(f"DEBUG: contents: {os.listdir(template_dir)}")
else:
    print(f"DEBUG: template_dir DOES NOT EXIST")

app = Flask(__name__, template_folder=template_dir)

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

        if mode == 'duplicate' or mode == 'similar':
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
            # Skip _trash_bin folder
            if '_trash_bin' in root.split(os.sep):
                continue
                
            for filename in filenames:
                if filename.startswith('.'): continue
                if os.path.splitext(filename)[1].lower() in valid_extensions:
                    files.append(os.path.join(root, filename))
        
        total = len(files)
        hashes = defaultdict(list)
        
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
                        elif mode == 'similar' and 'phash' in c_data:
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
        
        final_results = {} # Will hold the final response object
        
        if mode == 'duplicate':
            similar_groups = []
            for data in cached_results:
                img_info = ImageInfo(data)
                if img_info.phash:
                    hashes[str(img_info.phash)].append(img_info)

            for hash_str, img_list in hashes.items():
                if len(img_list) > 1:
                    group_data = [img.to_dict() for img in img_list]
                    group_data.sort(key=lambda x: (x['width'] * x['height'], x['size']), reverse=True)
                    similar_groups.append(group_data)
            
            final_results = {'similar_groups': similar_groups}
            state.progress_msg = f"扫描完成，发现 {len(similar_groups)} 组重复图片"
            
        elif mode == 'similar':
            # Calculate embeddings using CLIP model
            state.progress_msg = "正在加载 AI 模型 (首次运行可能需要几分钟下载)..."
            
            try:
                model = get_model()
                if model is None:
                     state.progress_msg = "错误: 请安装 sentence-transformers 以使用此功能 (pip install sentence-transformers)"
                     state.scanning = False
                     return

                # Collect valid images for embedding
                valid_images = []
                valid_paths = []
                
                total_imgs = len(cached_results)
                state.progress_msg = f"正在计算特征向量 (共 {total_imgs} 张)..."
                
                # Batch process for better performance? 
                # Or just loop. For < 1000 images, loop is fine.
                # But we need PIL images.
                
                for i, data in enumerate(cached_results):
                    if state.stop_flag: break
                    try:
                        img = Image.open(data['path']).convert('RGB')
                        valid_images.append(img)
                        valid_paths.append(data)
                    except:
                        pass
                    
                    if i % 10 == 0:
                        state.progress_msg = f"正在读取图片: {i}/{total_imgs}"
                
                if not valid_images:
                    state.scanning = False
                    state.progress_msg = "没有找到有效图片"
                    return

                state.progress_msg = "正在进行 AI 推理..."
                embeddings = model.encode(valid_images, convert_to_tensor=True, show_progress_bar=False)
                
                state.progress_msg = "正在聚类分析..."
                
                # Compute cosine similarity
                # util.cos_sim returns a matrix
                cosine_scores = util.cos_sim(embeddings, embeddings)
                
                # Cluster images
                # Threshold for "same object, different angle"
                # 0.85 is a good starting point for CLIP ViT-B-32
                threshold = 0.85 
                
                visited = set()
                similar_groups = []
                
                # Move tensor to cpu numpy for easy indexing
                cosine_scores_np = cosine_scores.cpu().numpy()
                
                for i in range(len(valid_paths)):
                    if i in visited: continue
                    
                    # Start a new group
                    current_group = [ImageInfo(valid_paths[i])]
                    visited.add(i)
                    
                    for j in range(i + 1, len(valid_paths)):
                        if j in visited: continue
                        
                        score = cosine_scores_np[i][j]
                        if score >= threshold:
                            current_group.append(ImageInfo(valid_paths[j]))
                            visited.add(j)
                    
                    if len(current_group) > 1:
                        group_data = [img.to_dict() for img in current_group]
                        # Sort by resolution/size
                        group_data.sort(key=lambda x: (x['width'] * x['height'], x['size']), reverse=True)
                        # Add similarity score info? Not needed for now.
                        similar_groups.append(group_data)

                final_results = {'similar_groups': similar_groups}
                state.progress_msg = f"扫描完成，发现 {len(similar_groups)} 组相似图片"

            except Exception as e:
                print(f"AI Error: {e}")
                state.progress_msg = f"AI 模型运行错误: {str(e)}"
            
        elif mode == 'blur':
            blur_images = []
            for data in cached_results:
                img_info = ImageInfo(data)
                if img_info.blur_score < 100: # Threshold for blur. <100 is usually blurry
                    blur_images.append(img_info.to_dict())
            
            # Sort by blur score (ascending = most blurry first)
            blur_images.sort(key=lambda x: x['blur_score'])
            final_results = {'blur_images': blur_images}
            state.progress_msg = f"扫描完成，发现 {len(blur_images)} 张模糊图片"

        state.results = final_results # Now state.results is a dict
        
    except Exception as e:
        state.progress_msg = f"错误: {str(e)}"
        print(e)
    finally:
        state.scanning = False

# --- Flask Routes ---

@app.route('/')
def index():
    try:
        return render_template('index.html')
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"Error: {e}", 500

import subprocess

# Installation State
install_state = {
    'installing': False,
    'message': '',
    'success': False,
    'error': None
}

def install_task(package_name):
    global install_state
    install_state['installing'] = True
    install_state['message'] = '正在初始化安装环境...'
    install_state['success'] = False
    install_state['error'] = None
    
    try:
        # Use --no-cache-dir to avoid cache issues? No, cache is good.
        # Use -v to get more verbose output for progress? Standard is fine.
        process = subprocess.Popen(
            [sys.executable, "-m", "pip", "install", package_name],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
            # Force unbuffered output for real-time updates
            env={**os.environ, "PYTHONUNBUFFERED": "1"} 
        )
        
        # Read output line by line
        for line in process.stdout:
            line = line.strip()
            if line:
                install_state['message'] = line
        
        process.wait()
        
        if process.returncode == 0:
            install_state['success'] = True
            install_state['message'] = '安装完成'
        else:
            install_state['success'] = False
            install_state['error'] = f'安装失败 (退出代码 {process.returncode})'
            install_state['message'] = f'安装失败 (退出代码 {process.returncode})'
            
    except Exception as e:
        install_state['success'] = False
        install_state['error'] = str(e)
        install_state['message'] = f'发生错误: {str(e)}'
    finally:
        install_state['installing'] = False

@app.route('/api/check_dependency', methods=['GET'])
def check_dependency():
    dep = request.args.get('name')
    if dep == 'sentence-transformers':
        try:
            import sentence_transformers
            return jsonify({'installed': True})
        except ImportError:
            return jsonify({'installed': False})
    return jsonify({'installed': False})

@app.route('/api/install_dependency', methods=['POST'])
def install_dependency():
    data = request.json
    dep = data.get('name')
    
    if dep == 'sentence-transformers':
        if install_state['installing']:
            return jsonify({'success': False, 'error': '安装正在进行中，请稍候...'})

        threading.Thread(target=install_task, args=(dep,), daemon=True).start()
        return jsonify({'success': True, 'status': 'started'})
            
    return jsonify({'success': False, 'error': '不支持安装该依赖'})

@app.route('/api/install_status', methods=['GET'])
def get_install_status():
    return jsonify(install_state)

@app.route('/api/scan', methods=['POST'])
def start_scan():
    data = request.json
    folder_path = data.get('path')
    # frontend sends booleans: find_similar, find_blur. For now we support one mode at a time or logic needs update.
    # The legacy code supported 'mode' param. The new frontend sends flags.
    # Let's adapt: if find_blur is true, run blur mode, else duplicate mode.
    # Or run both? The legacy logic ran one task.
    # For simplicity, let's stick to 'mode' if possible, or infer it.
    
    # Actually, the frontend code sends: path, delete_raw, find_similar, find_blur.
    # But startScan in frontend uses: body: JSON.stringify({ path: scanPath.value, ...settings })
    # So we receive find_similar=true/false.
    
    mode = 'duplicate'
    if data.get('find_similar'):
        mode = 'similar'
    elif data.get('find_blur'):
        mode = 'blur'
    
    # Default is duplicate if neither is true (though frontend radio should enforce one)
    
    if not folder_path or not os.path.exists(folder_path):
        return jsonify({'error': '无效的文件夹路径'}), 400
    
    if state.scanning:
        return jsonify({'error': '扫描正在进行中'}), 400
    
    state.stop_flag = False
    
    # Run scan in thread
    thread = threading.Thread(target=scan_task, args=(folder_path, mode))
    thread.daemon = True
    thread.start()
    
    return jsonify({'status': 'started'})

@app.route('/api/status')
def get_status():
    response = {
        'scanning': state.scanning,
        'message': state.progress_msg
    }
    if not state.scanning and state.results:
        response['results'] = state.results
    return jsonify(response)

@app.route('/api/image')
def get_image():
    path = request.args.get('path')
    if not path or not os.path.exists(path):
        return "File not found", 404
    return send_file(path)

import shutil

# ... (rest of imports)

@app.route('/api/delete', methods=['POST'])
def delete_images():
    data = request.json
    # Frontend sends { files: [...] }
    paths = data.get('files', [])
    deleted_files = []
    errors = []
    
    # Create trash directory if not exists
    # We will create a '_trash_bin' in the folder where the first file is located
    # Or in the scanned folder. We have state.folder_path
    
    trash_dir = os.path.join(state.folder_path, '_trash_bin')
    if not os.path.exists(trash_dir):
        try:
            os.makedirs(trash_dir)
        except Exception as e:
            return jsonify({'success': False, 'error': f'无法创建垃圾桶文件夹: {str(e)}'})
            
    for path in paths:
        try:
            if os.path.exists(path):
                # Move to trash_dir
                filename = os.path.basename(path)
                dest = os.path.join(trash_dir, filename)
                
                # Handle duplicate names in trash
                if os.path.exists(dest):
                    base, ext = os.path.splitext(filename)
                    counter = 1
                    while os.path.exists(dest):
                        dest = os.path.join(trash_dir, f"{base}_{counter}{ext}")
                        counter += 1
                
                shutil.move(path, dest)
                deleted_files.append(path)
        except Exception as e:
            errors.append({'path': path, 'error': str(e)})
            
    return jsonify({'success': True, 'deleted_count': len(deleted_files), 'deleted_files': deleted_files, 'errors': errors})

@app.route('/api/open_folder', methods=['POST'])
def open_folder():
    # Use tkinter to ask for directory
    try:
        import tkinter as tk
        from tkinter import filedialog
        
        # Create a hidden root window
        root = tk.Tk()
        root.withdraw()
        
        # Bring to front (hacky on Windows)
        root.attributes('-topmost', True)
        
        folder_path = filedialog.askdirectory()
        
        root.destroy()
        if not folder_path:
             return jsonify({'error': '未选择文件夹'})
        return jsonify({'path': folder_path})
    except Exception as e:
        return jsonify({'error': f'无法打开选择框: {str(e)}'})

def main():
    # Determine if running in a bundle or live
    # If we want to open browser
    url = 'http://127.0.0.1:5000'
    threading.Timer(1.5, lambda: webbrowser.open(url)).start()
    
    app.run(host='127.0.0.1', port=5000, debug=True, use_reloader=False)

if __name__ == '__main__':
    main()
