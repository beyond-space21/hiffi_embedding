.PHONY: preprocess-video extract-frames extract-audio run-worker run-api

extract-frames:
	@mkdir -p "$(TEMP_DIR)/$(VIDEO_ID)/frames"
	@ffmpeg -hide_banner -loglevel error -y -i "$(VIDEO_URL)" -vf "select='gt(scene,$(FRAME_SCENE_THRESHOLD))',fps=$(FRAME_EXTRACT_FPS)" -vsync vfr "$(TEMP_DIR)/$(VIDEO_ID)/frames/frame_%06d.png"

extract-audio:
	@mkdir -p "$(TEMP_DIR)/$(VIDEO_ID)"
	@ffmpeg -hide_banner -loglevel error -y -i "$(VIDEO_URL)" -vn -ac 1 -ar 48000 -c:a aac "$(TEMP_DIR)/$(VIDEO_ID)/audio.aac"

preprocess-video: extract-frames extract-audio
	@echo "preprocess done for $(VIDEO_ID)"

run-worker:
	@python3 worker.py

run-api:
	@cd app && uvicorn main:app --host 0.0.0.0 --port 8000 --reload
