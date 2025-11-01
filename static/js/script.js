document.getElementById('uploadForm').addEventListener('submit', async function(e) {
    e.preventDefault();
    
    const formData = new FormData(this);
    const resultSection = document.getElementById('resultSection');
    const emotionText = document.getElementById('emotionText');
    const confidenceText = document.getElementById('confidenceText');
    const resultImage = document.getElementById('resultImage');
    
    // Show loading state
    resultSection.style.display = 'block';
    emotionText.textContent = 'Processing...';
    confidenceText.textContent = '';
    resultImage.src = '';
    
    try {
        const response = await fetch('/detect_emotion', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (response.ok) {
            emotionText.textContent = `Emotion: ${data.emotion}`;
            confidenceText.textContent = `Confidence: ${data.confidence}%`;
            resultImage.src = data.image_path;
            
            // Add emotion-specific styling
            resultSection.className = 'result-section emotion-' + data.emotion.toLowerCase();
        } else {
            emotionText.textContent = 'Error: ' + data.error;
        }
    } catch (error) {
        emotionText.textContent = 'Error: ' + error.message;
    }
});