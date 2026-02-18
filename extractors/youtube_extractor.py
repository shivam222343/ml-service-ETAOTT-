import yt_dlp
import os
import whisper
import json
import uuid
import shutil
import subprocess

def setup_ffmpeg():
    """Ensure FFmpeg is in PATH, especially on Windows with local binaries."""
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("⚠️ Global FFmpeg not found, attempting to use imageio-ffmpeg binary...")
        try:
            import imageio_ffmpeg
            ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()
            ffmpeg_dir = os.path.dirname(ffmpeg_bin)
            
            # Create a proper "ffmpeg.exe" if it's named differently
            target_ffmpeg = os.path.join(ffmpeg_dir, "ffmpeg.exe")
            if not os.path.exists(target_ffmpeg):
                shutil.copy2(ffmpeg_bin, target_ffmpeg)
            
            if ffmpeg_dir not in os.environ["PATH"]:
                os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.environ["PATH"]
            return True
        except Exception as e:
            print(f"FFmpeg setup failed: {str(e)}")
            return False

def extract_youtube(video_url):
    """
    Extracts audio from a YouTube video and transcribes it using Whisper.
    Thread-safe implementation for parallel extractions.
    """
    # Create a unique job directory to allow parallel processing
    job_id = str(uuid.uuid4())[:8]
    base_temp = os.path.abspath("temp_youtube")
    job_dir = os.path.join(base_temp, job_id)
    os.makedirs(job_dir, exist_ok=True)
    
    # Ensure FFmpeg is available
    setup_ffmpeg()
    
    # yt-dlp options - output to specific job directory
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'outtmpl': os.path.join(job_dir, '%(id)s.%(ext)s'),
        'quiet': True,
        'no_warnings': True,
        'nocheckcertificate': True,
    }

    audio_path = None
    try:
        # 1. Extract metadata and download audio
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            print(f"📥 Downloading YouTube audio for job {job_id}...")
            info = ydl.extract_info(video_url, download=True)
            video_id = info['id']
            audio_path = os.path.join(job_dir, f"{video_id}.mp3")
            
            # Verify file exists (sometimes postprocessors fail silently)
            if not os.path.exists(audio_path):
                # Check for alternative extensions if ffmpeg failed to rename
                fallback_files = [f for f in os.listdir(job_dir) if f.startswith(video_id)]
                if fallback_files:
                    audio_path = os.path.join(job_dir, fallback_files[0])
                else:
                    raise FileNotFoundError(f"Audio file not found at {audio_path}")

            metadata = {
                "title": info.get('title'),
                "description": info.get('description'),
                "duration": info.get('duration'),
                "uploader": info.get('uploader'),
                "view_count": info.get('view_count'),
                "thumbnail": info.get('thumbnail'),
                "youtube_id": info.get('id')
            }

        # 2. Transcribe with Whisper
        from model_loader import model
        print(f"🎙️ Transcribing {job_id}: {metadata['title']}")
        result = model.transcribe(audio_path, fp16=False)

        return {
            "success": True,
            "metadata": metadata,
            "text": result["text"],
            "segments": result["segments"],
            "language": result.get("language", "en"),
            "summary": result["text"][:500] + "..." if len(result["text"]) > 500 else result["text"],
            "thumbnail_url": metadata.get("thumbnail"),
            "thumbnail_public_id": "youtube"
        }

    except Exception as e:
        print(f"❌ YouTube extraction error ({job_id}): {str(e)}")
        return {
            "success": False,
            "error": str(e)
        }
    finally:
        # Aggressive cleanup of only this job's directory
        try:
            if os.path.exists(job_dir):
                shutil.rmtree(job_dir, ignore_errors=True)
        except Exception as cleanup_err:
            print(f"⚠️ Cleanup failed for {job_id}: {cleanup_err}")
