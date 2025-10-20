# Snapchat Overlay Processor

This script applies overlay PNG files to their corresponding images and videos based on matching IDs in the filename. It creates new files with overlays applied without modifying the original files.

## Features

- **Automatic ID Matching**: Finds corresponding images/videos for each overlay based on ID in filename
- **High Quality Processing**: Preserves original image/video quality with configurable output settings
- **Progress Tracking**: Saves progress to JSON file for resumable processing
- **Batch Processing**: Processes all overlays in the memories directory
- **Verification Mode**: Check processing status without actually processing files

## Installation

Install the required dependencies:

```bash
pip install -r requirements_overlays.txt
```

Or install manually:

```bash
pip install Pillow opencv-python numpy
```

### Optional: FFmpeg for Video Audio Preservation

For video processing with audio preservation, install FFmpeg:

**Windows:**
```bash
# Using Chocolatey
choco install ffmpeg

# Or download from https://ffmpeg.org/download.html
```

**macOS:**
```bash
brew install ffmpeg
```

**Ubuntu/Debian:**
```bash
sudo apt install ffmpeg
```

**Note:** Without FFmpeg, videos will be processed without audio. The script will warn you if FFmpeg is not found.

## Usage

### Basic Usage

Process all overlays in the default memories directory:

```bash
python apply_overlays.py
```

### Advanced Usage

```bash
# Specify custom directories
python apply_overlays.py --memories-dir /path/to/memories --output-dir /path/to/output

# Resume from previous progress
python apply_overlays.py --resume

# Verify processing status without processing
python apply_overlays.py --verify

# Adjust quality settings
python apply_overlays.py --quality 98 --video-quality 20
```

### Command Line Options

- `--memories-dir`: Path to memories directory containing images, videos, and overlays (default: `memories`)
- `--output-dir`: Output directory for processed files with overlays (default: `memories_with_overlays`)
- `--progress-file`: Progress tracking file (default: `overlay_progress.json`)
- `--quality`: JPEG quality for output images, 1-100 (default: 95)
- `--video-quality`: Video quality for output videos, lower = better (default: 23)
- `--resume`: Resume from previous progress
- `--verify`: Verify processing without actually processing files

## How It Works

1. **Overlay Discovery**: Scans the `overlays/` directory for PNG files
2. **ID Extraction**: Parses filenames to extract media type and ID:
   - Format: `YYYY-MM-DD_HHMMSS_Type_ID_overlay.png`
   - Example: `2017-07-09_095148_Image_728C0018_overlay.png`
3. **Source Matching**: Finds corresponding source files:
   - Images: Looks in `images/` directory for matching ID
   - Videos: Looks in `videos/` directory for matching ID
4. **Processing**: Applies overlay to source file:
   - **Images**: Uses alpha blending with Pillow
   - **Videos**: Processes each frame with OpenCV
5. **Output**: Saves processed files to:
   - `images_with_overlays/` for processed images (with `_overlay` suffix)
   - `videos_with_overlays/` for processed videos (with `_overlay` suffix and preserved audio)

## File Structure

```
memories/
├── images/           # Original images
├── videos/           # Original videos
├── overlays/         # Overlay PNG files
└── ...

memories_with_overlays/
├── images_with_overlays/  # Processed images with overlays
│   └── 2017-07-09_095148_Image_728C0018_overlay.jpg
└── videos_with_overlays/  # Processed videos with overlays
    └── 2017-08-05_202639_Video_BCC64A59_overlay.mp4

overlay_progress.json      # Progress tracking file
```

## Progress Tracking

The script maintains a JSON file (`overlay_progress.json`) that tracks:

- **Processed files**: Successfully processed overlays
- **Failed files**: Failed processing attempts with error details
- **Resume capability**: Can resume from where it left off

## Quality Settings

- **Image Quality**: Controls JPEG compression (1-100, higher = better quality)
- **Video Quality**: Controls video compression (lower numbers = better quality)
  - Typical values: 18-28 (18 = very high quality, 28 = good quality)

## Error Handling

- **Missing Dependencies**: Clear error messages with installation instructions
- **Missing Source Files**: Logs which overlays don't have corresponding source files
- **Processing Errors**: Detailed error logging with timestamps
- **Resume Support**: Can retry failed processing by running the script again

## Examples

### Check what would be processed:

```bash
python apply_overlays.py --verify
```

### Process with high quality settings:

```bash
python apply_overlays.py --quality 98 --video-quality 18
```

### Resume processing after interruption:

```bash
python apply_overlays.py --resume
```

### Process specific directory:

```bash
python apply_overlays.py --memories-dir /path/to/snapchat/data --output-dir /path/to/processed
```

## Notes

- Original files are never modified
- Overlays are automatically resized to match source dimensions
- Alpha blending preserves transparency in overlays
- Video processing may take longer for longer videos
- Progress is saved after each successful processing
