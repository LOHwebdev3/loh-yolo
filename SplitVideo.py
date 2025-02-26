import os
import subprocess
import math

from utility import generate_unique_string


class SplitVideo:

    def __init__(self, **kwargs):
        """Initializes the ObjectCounter class for real-time object counting in video streams."""
        super().__init__(**kwargs)

    def get_video_duration(self,path=''):
        """Get duration (in seconds) of a video using ffprobe."""
        cmd = [
            'ffprobe', '-v', 'error',
            '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            path
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return float(result.stdout.strip())

    def hr(self,name='',vdo_path='',number_hr=1):
        if vdo_path == '':
            raise ValueError("Please provide a valid video file path.")

        if number_hr <= 0:
            raise ValueError("Number of hours should be greater than 0.")

        # Create an output folder with the same name as the video file (without extension)
        base_name = os.path.splitext(os.path.basename(vdo_path))[0]
        output_dir = f'tmp/split_vdo/{name}'

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created folder: {output_dir}")
        else:
            print(f"Folder already exists: {output_dir}")

        # Get the total duration of the video
        total_duration = self.get_video_duration(path=vdo_path)
        segment_duration = number_hr*3600  # 1 hour in seconds
        num_segments = math.ceil(total_duration / segment_duration)

        # Loop to split the video into 1-hour segments
        for i in range(num_segments):
            start_time = i * segment_duration
            output_file = os.path.join(output_dir, f"{base_name}_part_{(i + 1):02d}.mp4")

            # Build FFmpeg command with lower bitrate, CQ, and reduced resolution
            cmd = (
                f"ffmpeg -hwaccel cuda -ss {start_time:.2f} -i {vdo_path} "
                f"-t {segment_duration:.2f} -vf scale=1280:720 "
                f"-c:v h264_nvenc -rc:v vbr -cq 28 -b:v 2M "
                f"-c:a aac -b:a 128k {output_file} -y"
            )

            print(f"Executing: {cmd}")
            subprocess.run(cmd, shell=True)

        print("Video splitting completed.")
