#!/usr/bin/env python3
"""
Snapchat Overlay Processor

This script applies overlay PNG files to their corresponding images and videos
based on matching IDs in the filename. It creates new files with overlays
applied without modifying the original files.
"""

import os
import re
import json
import time
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional

# Optional imports - will be checked in check_dependencies()
try:
    from PIL import Image, ImageDraw
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == '__main__':
    # Import main and run it
    pass  # Will be replaced at the end


# ============================================================================
# Main Function - Script Entry Point
# ============================================================================

def main():
    """Main entry point - parses arguments and orchestrates the overlay processing."""
    import argparse

    parser = argparse.ArgumentParser(description='Apply overlays to Snapchat memories')
    parser.add_argument('--memories-dir', default='memories',
                        help='Path to memories directory containing images, videos, and overlays')
    parser.add_argument('--output-dir', default='memories_with_overlays',
                        help='Output directory for processed files with overlays')
    parser.add_argument('--progress-file', default='overlay_progress.json',
                        help='Progress tracking file')
    parser.add_argument('--quality', type=int, default=95,
                        help='JPEG quality for output images (1-100, default: 95)')
    parser.add_argument('--video-quality', type=int, default=23,
                        help='Video quality for output videos (lower = better, default: 23)')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from previous progress')
    parser.add_argument('--verify', action='store_true',
                        help='Verify processing without actually processing files')

    args = parser.parse_args()

    # Check dependencies before starting
    check_dependencies()

    # Create processor instance
    processor = OverlayProcessor(
        memories_dir=args.memories_dir,
        output_dir=args.output_dir,
        progress_file=args.progress_file,
        image_quality=args.quality,
        video_quality=args.video_quality
    )

    # Run in verification mode or processing mode
    if args.verify:
        print("Verifying overlay processing...")
        results = processor.verify_processing()

        print(f"\nVerification Results:")
        print(f"{'='*60}")
        print(f"Total overlays: {results['total_overlays']}")
        print(f"Processed: {results['processed']}")
        print(f"Missing source files: {len(results['missing_sources'])}")
        print(f"Failed: {len(results['failed'])}")
        print(f"{'='*60}\n")

        if results['missing_sources']:
            print("Missing source files:")
            for item in results['missing_sources'][:10]:
                print(f"  - {item['overlay_file']} -> {item['source_file']}")
            if len(results['missing_sources']) > 10:
                print(f"  ... and {len(results['missing_sources']) - 10} more")

        if results['failed']:
            print("\nFailed processing:")
            for item in results['failed'][:10]:
                print(f"  - {item['overlay_file']} ({item['error']})")
            if len(results['failed']) > 10:
                print(f"  ... and {len(results['failed']) - 10} more")
    else:
        # Process all overlays
        processor.process_all_overlays(resume=args.resume)


# ============================================================================
# Dependency Checking
# ============================================================================

def check_dependencies():
    """Check for required dependencies and prompt user."""
    import sys

    missing_deps = []
    
    if not PIL_AVAILABLE:
        missing_deps.append("Pillow (PIL)")
    
    if not CV2_AVAILABLE:
        missing_deps.append("opencv-python")
    
    if not NUMPY_AVAILABLE:
        missing_deps.append("numpy")

    if missing_deps:
        print("\n" + "="*70)
        print("MISSING DEPENDENCIES")
        print("="*70)
        print("\nThe following required packages are not installed:\n")

        for dep in missing_deps:
            print(f"  • {dep}")

        print("\nInstallation instructions:")
        print("  pip install Pillow opencv-python numpy")
        print("\nExiting. Please install the dependencies and run the script again.")
        print("="*70)
        sys.exit(1)
    else:
        print("\n" + "="*70)
        print("All dependencies found!")
        print("  [OK] Pillow: Image processing")
        print("  [OK] OpenCV: Video processing")
        print("  [OK] NumPy: Array operations")
        print("="*70 + "\n")


# ============================================================================
# OverlayProcessor Class - Main Orchestrator
# ============================================================================

class OverlayProcessor:
    """Process overlays and apply them to images and videos.

    Methods are organized in execution order - read from top to bottom
    to follow the flow of a typical processing session.
    """

    # ========================================================================
    # Initialization
    # ========================================================================

    def __init__(self, memories_dir: str, output_dir: str, progress_file: str, 
                 image_quality: int = 95, video_quality: int = 23):
        """Initialize the processor with configuration."""
        self.memories_dir = Path(memories_dir)
        self.output_dir = Path(output_dir)
        self.progress_file = progress_file
        self.image_quality = image_quality
        self.video_quality = video_quality
        self.progress = self._load_progress()

        # Create output directories
        self.output_dir.mkdir(exist_ok=True)
        (self.output_dir / "images_with_overlays").mkdir(exist_ok=True)
        (self.output_dir / "videos_with_overlays").mkdir(exist_ok=True)

    def _load_progress(self) -> Dict:
        """Load processing progress from JSON file."""
        if os.path.exists(self.progress_file):
            with open(self.progress_file, 'r') as f:
                return json.load(f)
        return {'processed': {}, 'failed': {}}

    def _save_progress(self):
        """Save processing progress to JSON file."""
        with open(self.progress_file, 'w') as f:
            json.dump(self.progress, f, indent=2)

    # ========================================================================
    # Main Processing Flow
    # ========================================================================

    def process_all_overlays(self, resume: bool = True):
        """Process all overlays with progress tracking."""
        # Step 1: Find all overlay files
        overlays = self._find_overlay_files()
        
        if not overlays:
            print("No overlay files found!")
            return

        # Step 2: Calculate what needs to be processed
        total = len(overlays)
        already_processed = len([o for o in overlays if o['overlay_file'] in self.progress['processed']])
        to_process = total - already_processed

        print(f"\nTotal overlays: {total}")
        print(f"Already processed: {already_processed}")
        print(f"To process: {to_process}\n")

        if to_process == 0:
            print("All overlays already processed!")
            return

        # Step 3: Process each overlay
        processed_count = 0
        failed_count = 0
        skipped_count = 0

        for i, overlay in enumerate(overlays, 1):
            overlay_file = overlay['overlay_file']

            if resume and overlay_file in self.progress['processed']:
                print(f"[{i}/{total}] Skipping {overlay_file} (already processed)")
                skipped_count += 1
                continue

            print(f"[{i}/{total}] Processing {overlay_file}...", end=" ")

            success, message = self.process_overlay(overlay)
            print(message)

            if success:
                processed_count += 1
            else:
                failed_count += 1

        # Step 4: Print summary
        print(f"\n{'='*60}")
        print(f"Processing complete!")
        print(f"Processed: {processed_count}")
        print(f"Failed: {failed_count}")
        print(f"Skipped: {skipped_count}")
        print(f"Total: {total}")
        print(f"{'='*60}\n")

        if failed_count > 0:
            print(f"Failed processing are tracked in {self.progress_file}")
            print("Run the script again to retry failed processing.\n")

    def _find_overlay_files(self) -> List[Dict]:
        """Find all overlay files and extract their metadata."""
        overlays_dir = self.memories_dir / "overlays"
        
        if not overlays_dir.exists():
            return []

        overlays = []
        for overlay_file in overlays_dir.glob("*.png"):
            # Parse filename: YYYY-MM-DD_HHMMSS_Type_ID_overlay.png
            match = re.match(r'(\d{4}-\d{2}-\d{2})_(\d{6})_(Image|Video)_([A-F0-9]+)_overlay\.png', 
                           overlay_file.name)
            if match:
                date_part, time_part, media_type, file_id = match.groups()
                
                # Find corresponding source file
                source_file = self._find_source_file(media_type, file_id)
                
                overlays.append({
                    'overlay_file': overlay_file.name,
                    'overlay_path': overlay_file,
                    'media_type': media_type.lower(),
                    'file_id': file_id,
                    'source_file': source_file,
                    'date': f"{date_part}_{time_part}"
                })

        return overlays

    def _find_source_file(self, media_type: str, file_id: str) -> Optional[Path]:
        """Find the source file (image or video) that matches the overlay ID."""
        if media_type.lower() == 'image':
            source_dir = self.memories_dir / "images"
            pattern = f"*{file_id}*.jpg"
        else:  # video
            source_dir = self.memories_dir / "videos"
            pattern = f"*{file_id}*.mp4"

        if not source_dir.exists():
            return None

        matches = list(source_dir.glob(pattern))
        return matches[0] if matches else None

    def process_overlay(self, overlay: Dict) -> Tuple[bool, str]:
        """Process a single overlay and apply it to the source file."""
        overlay_file = overlay['overlay_file']
        source_file = overlay['source_file']

        # Check if already processed
        if overlay_file in self.progress['processed']:
            return True, "Already processed"

        # Check if source file exists
        if not source_file or not source_file.exists():
            return self._record_failure(overlay_file, f"Source file not found: {source_file}")

        try:
            if overlay['media_type'] == 'image':
                return self._process_image_overlay(overlay)
            else:  # video
                return self._process_video_overlay(overlay)
        except Exception as e:
            return self._record_failure(overlay_file, f"Processing error: {str(e)}")

    def _process_image_overlay(self, overlay: Dict) -> Tuple[bool, str]:
        """Process an image with overlay."""
        overlay_file = overlay['overlay_file']
        source_file = overlay['source_file']
        
        # Load source image
        source_img = Image.open(source_file)
        if source_img.mode != 'RGBA':
            source_img = source_img.convert('RGBA')
        
        # Load overlay
        overlay_img = Image.open(overlay['overlay_path'])
        if overlay_img.mode != 'RGBA':
            overlay_img = overlay_img.convert('RGBA')
        
        # Resize overlay to match source image
        overlay_resized = overlay_img.resize(source_img.size, Image.Resampling.LANCZOS)
        
        # Composite overlay onto source
        result_img = Image.alpha_composite(source_img, overlay_resized)
        
        # Convert back to RGB for JPEG output
        if result_img.mode == 'RGBA':
            # Create white background
            background = Image.new('RGB', result_img.size, (255, 255, 255))
            background.paste(result_img, mask=result_img.split()[-1])  # Use alpha channel as mask
            result_img = background
        
        # Create output filename with overlay suffix
        output_filename = self._create_output_filename(source_file.name, 'image')
        output_file = self.output_dir / "images_with_overlays" / output_filename
        result_img.save(output_file, 'JPEG', quality=self.image_quality, optimize=True)
        
        # Record success
        self.progress['processed'][overlay_file] = {
            'source_file': source_file.name,
            'output_file': output_file.name,
            'media_type': 'image',
            'timestamp': datetime.now().isoformat()
        }
        self._save_progress()
        
        return True, "Processed successfully"

    def _process_video_overlay(self, overlay: Dict) -> Tuple[bool, str]:
        """Process a video with overlay."""
        overlay_file = overlay['overlay_file']
        source_file = overlay['source_file']
        
        # Load overlay image
        overlay_img = cv2.imread(str(overlay['overlay_path']), cv2.IMREAD_UNCHANGED)
        if overlay_img is None:
            return self._record_failure(overlay_file, "Could not load overlay image")
        
        # Open video
        cap = cv2.VideoCapture(str(source_file))
        if not cap.isOpened():
            return self._record_failure(overlay_file, "Could not open source video")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Resize overlay to match video dimensions
        overlay_resized = cv2.resize(overlay_img, (width, height))
        
        # Create output filename with overlay suffix
        output_filename = self._create_output_filename(source_file.name, 'video')
        output_file = self.output_dir / "videos_with_overlays" / output_filename
        
        # Use FFmpeg for video processing to preserve audio
        temp_video = self.output_dir / "videos_with_overlays" / f"temp_{overlay_file}.mp4"
        
        # First, process video frames with overlay
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(temp_video), fourcc, fps, (width, height))
        
        if not out.isOpened():
            cap.release()
            return self._record_failure(overlay_file, "Could not create temporary video")
        
        # Process each frame
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert frame to RGBA for alpha blending
            frame_rgba = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
            
            # Apply overlay with alpha blending
            if overlay_resized.shape[2] == 4:  # Has alpha channel
                alpha = overlay_resized[:, :, 3] / 255.0
                alpha = np.stack([alpha] * 3, axis=2)
                
                # Blend overlay onto frame
                frame_rgba[:, :, :3] = (1 - alpha) * frame_rgba[:, :, :3] + alpha * overlay_resized[:, :, :3]
            else:
                # No alpha channel, just overlay
                frame_rgba[:, :, :3] = overlay_resized[:, :, :3]
            
            # Convert back to BGR for video writer
            frame_bgr = cv2.cvtColor(frame_rgba, cv2.COLOR_BGRA2BGR)
            out.write(frame_bgr)
            
            frame_count += 1
            if frame_count % 100 == 0:
                print(f"  Frame {frame_count}/{total_frames}", end="\r")
        
        # Cleanup video capture and writer
        cap.release()
        out.release()
        
        # Use FFmpeg to combine video with original audio
        try:
            import subprocess
            
            # Check if FFmpeg is available
            ffmpeg_cmd = self._find_ffmpeg()
            if ffmpeg_cmd:
                # Combine video with original audio
                cmd = [
                    ffmpeg_cmd,
                    '-i', str(source_file),  # Original video with audio
                    '-i', str(temp_video),   # Processed video without audio
                    '-c:v', 'libx264',       # Video codec
                    '-c:a', 'aac',           # Audio codec
                    '-map', '0:a',           # Use audio from original
                    '-map', '1:v',           # Use video from processed
                    '-crf', str(self.video_quality),  # Video quality
                    '-y',                    # Overwrite output file
                    str(output_file)
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
                
                if result.returncode == 0:
                    # Success - remove temp file
                    temp_video.unlink()
                else:
                    # FFmpeg failed - copy temp file as fallback
                    temp_video.rename(output_file)
                    print(f"  Warning: Audio preservation failed, using video without audio")
            else:
                # No FFmpeg - just rename temp file
                temp_video.rename(output_file)
                print(f"  Warning: FFmpeg not found, video will have no audio")
                
        except Exception as e:
            # Fallback - just rename temp file
            temp_video.rename(output_file)
            print(f"  Warning: Audio processing failed ({str(e)}), using video without audio")
        
        # Record success
        self.progress['processed'][overlay_file] = {
            'source_file': source_file.name,
            'output_file': output_file.name,
            'media_type': 'video',
            'timestamp': datetime.now().isoformat()
        }
        self._save_progress()
        
        return True, f"Processed successfully ({frame_count} frames)"

    def _record_failure(self, overlay_file: str, error_msg: str) -> Tuple[bool, str]:
        """Record a failed processing attempt."""
        if overlay_file not in self.progress['failed']:
            self.progress['failed'][overlay_file] = {'count': 0, 'errors': []}

        self.progress['failed'][overlay_file]['count'] += 1
        self.progress['failed'][overlay_file]['errors'].append({
            'timestamp': datetime.now().isoformat(),
            'error': error_msg
        })
        self._save_progress()

        return False, f"Error: {error_msg}"

    # ========================================================================
    # Verification Flow
    # ========================================================================

    def verify_processing(self) -> Dict:
        """Verify all overlays have been processed."""
        overlays = self._find_overlay_files()

        results = {
            'total_overlays': len(overlays),
            'processed': 0,
            'missing_sources': [],
            'failed': []
        }

        for overlay in overlays:
            overlay_file = overlay['overlay_file']
            
            if overlay_file in self.progress['processed']:
                results['processed'] += 1
            elif not overlay['source_file'] or not overlay['source_file'].exists():
                results['missing_sources'].append({
                    'overlay_file': overlay_file,
                    'source_file': str(overlay['source_file'])
                })
            elif overlay_file in self.progress['failed']:
                results['failed'].append({
                    'overlay_file': overlay_file,
                    'error': self.progress['failed'][overlay_file]['errors'][-1]['error']
                })

        return results

    def _create_output_filename(self, source_filename: str, media_type: str) -> str:
        """Create output filename with overlay suffix."""
        # Remove extension
        name_without_ext = source_filename.rsplit('.', 1)[0]
        extension = source_filename.rsplit('.', 1)[1] if '.' in source_filename else ('jpg' if media_type == 'image' else 'mp4')
        
        # Add overlay suffix
        return f"{name_without_ext}_overlay.{extension}"

    def _find_ffmpeg(self) -> Optional[str]:
        """Find FFmpeg executable."""
        import shutil
        
        # Check if FFmpeg is in PATH
        ffmpeg_path = shutil.which('ffmpeg')
        if ffmpeg_path:
            return ffmpeg_path
        
        # Check common installation paths
        common_paths = [
            'C:\\ffmpeg\\bin\\ffmpeg.exe',
            'C:\\Program Files\\ffmpeg\\bin\\ffmpeg.exe',
            'C:\\Program Files (x86)\\ffmpeg\\bin\\ffmpeg.exe',
            '/usr/bin/ffmpeg',
            '/usr/local/bin/ffmpeg',
            '/opt/homebrew/bin/ffmpeg'
        ]
        
        for path in common_paths:
            if os.path.exists(path):
                return path
        
        return None


# ============================================================================
# Script Execution
# ============================================================================

if __name__ == '__main__':
    main()
