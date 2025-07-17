import logging, shutil, subprocess

def setup_logging():
    logging.basicConfig(filename="logs/pilot.log", level=logging.INFO)

def backup_and_reset(block_id):
    shutil.make_archive(f"backup/block_{block_id}", 'zip', "logs/")
    subprocess.run(["git", "reset", "--hard", "HEAD"])
