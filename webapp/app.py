from flask import Flask, request, render_template, redirect, url_for
import os
from inference import predict_on_video

UPLOAD = "/Users/ankushaggarwal/Desktop/fitness-pose-project/data/sample_videos"
os.makedirs(UPLOAD, exist_ok=True)
app = Flask(__name__)

@app.route("/", methods=["GET","POST"])
def index():
    if request.method == "POST":
        f = request.files["file"]
        fname = os.path.join(UPLOAD, f.filename)
        f.save(fname)
        preds = predict_on_video(fname)
        
        if preds:
            labels = [p["label"] for p in preds]
            from collections import Counter
            c = Counter(labels)
            top_label, count = c.most_common(1)[0]
        else:
            top_label = "no_pose_detected"
        return render_template("result.html", predictions=preds, top_label=top_label)
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
