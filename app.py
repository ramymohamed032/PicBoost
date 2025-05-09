from flask import Flask, render_template, request
import os
import cv2
from enhancement import *
import uuid

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/', methods=['GET', 'POST'])
def index():
    output_path = None
    psnr = mse = None
    method_used = None
    original_path = None  # Initialize the original path variable

    if request.method == 'POST':
        file = request.files['image']
        method = request.form['method']
        if file:
            filename = str(uuid.uuid4()) + ".png"
            path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(path)
            
            image = cv2.imread(path)  # Make sure to read the image after saving it

            if image is None:
                return render_template('index.html', error="فشل في قراءة الصورة. تأكد من أن الملف صورة صالحة.")

            original_path = os.path.join(app.config['UPLOAD_FOLDER'], 'original_' + filename)
            cv2.imwrite(original_path, image)  # Save the original image to the server

            method_used = method
            invert = request.form.get('invert') == 'on'

            if invert:
                result = cv2.bitwise_not(image)  # Invert the image if the user selected invert
            else:
                if method == 'hist_eq':
                    result = histogram_equalization(image)
                elif method == 'average':
                    result = image_averaging(image)
                elif method == 'lowpass':
                    result = low_pass_filter(image)
                elif method == 'sharpen':
                    result = sharpen_image(image)
                elif method == 'gaussian':
                    result = gaussian_smoothing(image)
                elif method == 'saltpepper':
                    result = remove_salt_and_pepper(image)
                else:
                    result = image  # No enhancement applied

            mse, psnr = compute_metrics(image.astype(np.float64), result.astype(np.float64))
            output_path = os.path.join(app.config['UPLOAD_FOLDER'], 'out_' + filename)
            cv2.imwrite(output_path, result)  # Save the enhanced image

    return render_template('index.html', output=output_path, psnr=psnr, mse=mse, method=method_used, original=original_path)

if __name__ == '__main__':
    app.run(debug=True)
