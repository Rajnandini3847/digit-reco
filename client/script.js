const canvas = document.getElementById('canvas');
        const debugCanvas = document.getElementById('debug-canvas');
        const ctx = canvas.getContext('2d');
        const debugCtx = debugCanvas.getContext('2d');
        const resultBox = document.getElementById('result');

        // Set white background
        ctx.fillStyle = 'white';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        debugCtx.fillStyle = 'white';
        debugCtx.fillRect(0, 0, debugCanvas.width, debugCanvas.height);

        let isDrawing = false;
        let lastX = 0;
        let lastY = 0;

        function draw(e) {
            if (!isDrawing) return;
            
            const rect = canvas.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;

            ctx.beginPath();
            ctx.moveTo(lastX, lastY);
            ctx.lineTo(x, y);
            ctx.strokeStyle = 'black';
            ctx.lineWidth = 15;
            ctx.lineCap = 'round';
            ctx.stroke();

            lastX = x;
            lastY = y;
        }

        function updateDebugCanvas() {
            // Create temporary canvas for resizing
            const tempCanvas = document.createElement('canvas');
            tempCanvas.width = 28;
            tempCanvas.height = 28;
            const tempCtx = tempCanvas.getContext('2d');

            // Set white background
            tempCtx.fillStyle = 'white';
            tempCtx.fillRect(0, 0, 28, 28);

            // Draw resized image
            tempCtx.drawImage(canvas, 0, 0, 28, 28);

            // Get image data and process it
            const imageData = tempCtx.getImageData(0, 0, 28, 28);
            const data = imageData.data;

            // Convert to grayscale and invert colors
            for (let i = 0; i < data.length; i += 4) {
                const avg = (data[i] + data[i + 1] + data[i + 2]) / 3;
                data[i] = data[i + 1] = data[i + 2] = 255 - avg;
            }

            // Put processed image data back
            tempCtx.putImageData(imageData, 0, 0);

            // Draw to debug canvas
            debugCtx.drawImage(tempCanvas, 0, 0);
            
            return tempCanvas.toDataURL();
        }

        canvas.addEventListener('mousedown', (e) => {
            isDrawing = true;
            const rect = canvas.getBoundingClientRect();
            lastX = e.clientX - rect.left;
            lastY = e.clientY - rect.top;
        });

        canvas.addEventListener('mousemove', draw);
        canvas.addEventListener('mouseup', () => isDrawing = false);
        canvas.addEventListener('mouseout', () => isDrawing = false);

        document.getElementById('clear-btn').addEventListener('click', () => {
            ctx.fillStyle = 'white';
            ctx.fillRect(0, 0, canvas.width, canvas.height);
            debugCtx.fillStyle = 'white';
            debugCtx.fillRect(0, 0, debugCanvas.width, debugCanvas.height);
            resultBox.innerHTML = '';
        });

        document.getElementById('predict-btn').addEventListener('click', async () => {
            try {
                resultBox.innerHTML = 'Processing...';
                
                // Update debug canvas and get processed image
                const processedImage = updateDebugCanvas();
                
                const response = await fetch('http://127.0.0.1:5000/predict', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        image: processedImage
                    })
                });

                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }

                const data = await response.json();
                resultBox.innerHTML = `Predicted Digit: ${data.prediction}, Confidence: ${Math.round(data.confidence * 100)}%`;
            } catch (error) {
                console.error('Error:', error);
                resultBox.innerHTML = `Error: ${error.message}`;
            }
        });