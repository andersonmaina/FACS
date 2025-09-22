from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image, ImageDraw
import io
import base64

from model import predict

app = Flask(__name__)
CORS(app)

#global variables
threshold = 0.02
diagnosis = ''

@app.route('/api/process-annotation', methods=['POST'])
def process_annotation():
    try:
        if request.content_type and request.content_type.startswith('multipart/form-data'):
            source = request.form.get('source', 'telegram')
        else:
            data = request.get_json()
            source = data.get('source', 'pc') if data else 'pc'

        if source == "pc":
            if not data:
                return jsonify({"error": "No JSON data provided"}), 400
            img_data = data.get('image')
            bbox = data.get('coordinates')
            category = data.get('category')
            view = data.get('view')

            #Decode base64 image
            header, encoded = img_data.split(',', 1)
            img_bytes = base64.b64decode(encoded)
            img = Image.open(io.BytesIO(img_bytes))

            cropped = img.crop((
                bbox['x'],
                bbox['y'],
                bbox['x'] + bbox['width'],
                bbox['y'] + bbox['height']
            ))
            print(f"\nParameters:[Category-{category}, View-{view}, Annotations-{bbox}]\n")
            print("Processing...\n")
            error = predict(cropped, view, category)
            label = get_label(error)
            confidence = 100 - (100 * float(error))
            diagnosis = build_diagnosis(error, confidence)
            
            draw = ImageDraw.Draw(img)
            draw.rectangle(
                [bbox['x'], bbox['y'], bbox['x'] + bbox['width'], bbox['y'] + bbox['height']],
                outline="red", width=3
            )

            # Re-encode processed images to base64
            buffered = io.BytesIO()
            cropped.save(buffered, format="PNG")
            processed_b64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
            processed_data_url = f"data:image/png;base64,{processed_b64}"

            print(f"Results: Error-{error}, Comment-{label}, Confidence-{confidence}, Diagnosis-{diagnosis}\n")
            print(f"Rendering results to: {source} \n")
            
            return jsonify({
                "processed_image": processed_data_url,
                "category": category,
                "comment": label,
                "error": error,
                "confidence": confidence,
                "threshold": threshold,
                "diagnosis": diagnosis
            })

        elif source == "telegram":
            category = request.form.get('category')
            view = request.form.get('view')
            source_field = request.form.get('source')
            image_file = request.files.get('image')
            
            if not image_file:
                return jsonify({"error": "No image file provided"}), 400
                
            if not category or not view:
                return jsonify({"error": "Missing category or view"}), 400

            try:
                img = Image.open(image_file)
                
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                    
            except Exception as e:
                return jsonify({"error": f"Invalid image file: {str(e)}"}), 400

            print(f"\nParameters:[Category-{category}, View-{view}, Image size-{img.size}\n")
            print("Processing...\n")
            
            error = predict(img, view, category)
            label = get_label(error)
            confidence = 100 - (100 * float(error))
            diagnosis = build_diagnosis(error, confidence)
            
            # Convert image back to base64 for response
            buffered = io.BytesIO()
            img.save(buffered, format="PNG")
            processed_b64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
            processed_data_url = f"data:image/png;base64,{processed_b64}"

            response_data = {
                "processed_image": processed_data_url,
                "category": category,
                "view": view,
                "comment": label,
                "error": error,
                "confidence": confidence,
                "threshold": threshold,
                "diagnosis": diagnosis,
                "source": "telegram"
            }

            print(f"Results: Error-{error}, Comment-{label}, Confidence-{confidence}, Diagnosis-{diagnosis}\n")
            print(f"Rendering results to: {source} \n")
            
            return jsonify(response_data)
        else:
            return jsonify({"error": "Unsupported source"}), 400

    except Exception as e:
        print(f"Unexpected error: {e}")
        return jsonify({"error": str(e)}), 500

def get_label(error):
    label = "⚠ Anomaly detected" if error > threshold else "✅ Normal structure"
    return label

#RUBRIC      
def build_diagnosis(error, confidence):
    if confidence < 100 and confidence > 98 and error > threshold:
        diagnosis = 'High probability that there is an anomaly in the structure.'
    elif confidence < 98 and confidence > 96 and error > threshold:
        diagnosis = 'My analysis concludes there could be an anomaly in the structure, but either the structure is not well annotated or the image could be distorted, small, unclear or bad hence the uncertainty.'
    elif confidence < 96 and confidence > 92 and error > threshold:
        diagnosis = 'High possibility that anomaly is false psoitive and image is greatly distorted or irrelevant. Please check the image or the annotation and try again. If results are same, consult an expert.'
    elif confidence < 92 and error > threshold:
        diagnosis = 'Please upload a good ultrasound scan to obtain diagnosis. I cannot recognize the image nor the outlined structure.'
    elif confidence < 100 and confidence > 98 and error < threshold:
        diagnosis = 'Healthy structure detected. Annotation is correct OR model partially detects healthy area.'
    
    
    diagnosis += " THIS IS NOT PROFESSIONAL MEDICAL ADVICE. LIAISE WITH AN EXPERT"
    
    return diagnosis

if __name__ == '__main__':
    app.run(debug=True, port=5000)