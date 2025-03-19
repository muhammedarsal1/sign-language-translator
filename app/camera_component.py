import streamlit.components.v1 as components

def camera_input():
    """JavaScript-based camera input for real-time video."""
    camera_html = """
    <script>
        let video = document.createElement('video');
        let canvas = document.createElement('canvas');
        let context = canvas.getContext('2d');
        let captureButton = document.createElement('button');
        let capturedImage = document.createElement('img');

        video.setAttribute('autoplay', '');
        video.setAttribute('playsinline', '');
        video.style.width = '100%';
        video.style.height = 'auto';

        captureButton.innerText = 'Capture Image';
        captureButton.style.margin = '10px';

        captureButton.onclick = function() {
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            context.drawImage(video, 0, 0, canvas.width, canvas.height);
            capturedImage.src = canvas.toDataURL('image/png');
            window.parent.postMessage(capturedImage.src, "*");
        };

        navigator.mediaDevices.getUserMedia({ video: { facingMode: "environment" } })
            .then(stream => {
                video.srcObject = stream;
                document.getElementById('camera-container').appendChild(video);
                document.getElementById('camera-container').appendChild(captureButton);
            })
            .catch(error => {
                document.getElementById("camera-container").innerText = "🚨 Camera not accessible. Please enable permissions.";
            });
    </script>
    <div id="camera-container"></div>
    """
    return components.html(camera_html, height=500).value
