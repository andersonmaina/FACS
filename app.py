from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image, ImageDraw
import io
import base64
import traceback
import logging

from model import predict

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Enhanced CORS configuration
CORS(app, resources={
    r"/api/*": {
        "origins": ["https://facsai.vercel.app", "http://localhost:3000", "http://localhost:5000"],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type"],
        "supports_credentials": True,
        "max_age": 3600
    }
})

# Global variables
threshold = 0.02
diagnosis = ''

@app.route('/api/process-annotation', methods=['POST', 'OPTIONS'])
def process_annotation():
    """
    Process annotation endpoint - handles both PC (JSON) and Telegram (form-data) sources
    """
    try:
        logger.info(f"Request received - Content-Type: {request.content_type}")
        
        # Determine source
        if request.content_type and request.content_type.startswith('multipart/form-data'):
            source = request.form.get('source', 'telegram')
            logger.info(f"Form data source: {source}")
        else:
            data = request.get_json(silent=True)
            if data is None:
                logger.error("No JSON data provided")
                return jsonify({
                    "error": "No JSON data provided",
                    "details": "Request must be JSON or form-data"
                }), 400
            source = data.get('source', 'pc')
            logger.info(f"JSON source: {source}")

        if source == "pc":
            # Handle PC source (JSON with base64 image)
            logger.info("Processing PC source request")
            
            if data is None:
                logger.error("Data is None for PC source")
                return jsonify({
                    "error": "No JSON data provided",
                    "details": "PC source requires JSON payload"
                }), 400
            
            # Extract and validate required fields
            img_data = data.get('image')
            bbox = data.get('coordinates')
            category = data.get('category')
            view = data.get('view')
            
            # Validate required fields
            if not all([img_data, bbox, category, view]):
                missing_fields = [
                    field for field, value in [
                        ('image', img_data),
                        ('coordinates', bbox),
                        ('category', category),
                        ('view', view)
                    ] if not value
                ]
                logger.error(f"Missing required fields: {missing_fields}")
                return jsonify({
                    "error": "Missing required fields",
                    "missing_fields": missing_fields
                }), 400

            try:
                # Decode base64 image
                if ',' in img_data:
                    header, encoded = img_data.split(',', 1)
                else:
                    encoded = img_data
                
                img_bytes = base64.b64decode(encoded)
                img = Image.open(io.BytesIO(img_bytes))
                logger.info(f"Image decoded successfully: {img.size}")
                
            except Exception as e:
                logger.error(f"Failed to decode image: {str(e)}")
                return jsonify({
                    "error": "Invalid base64 image",
                    "details": str(e)
                }), 400

            try:
                # Validate and crop image
                cropped = img.crop((
                    bbox['x'],
                    bbox['y'],
                    bbox['x'] + bbox['width'],
                    bbox['y'] + bbox['height']
                ))
                logger.info(f"Image cropped: {cropped.size}")
                
            except Exception as e:
                logger.error(f"Failed to crop image: {str(e)}")
                return jsonify({
                    "error": "Invalid coordinates",
                    "details": str(e)
                }), 400

            logger.info(f"Parameters:[Category-{category}, View-{view}, Annotations-{bbox}]")
            logger.info("Processing image...")
            
            try:
                # Run prediction
                error = predict(cropped, view, category)
                label = get_label(error)
                confidence = 100 - (100 * float(error))
                diagnosis = build_diagnosis(error, confidence)
                
                logger.info(f"Prediction successful: Error-{error}, Confidence-{confidence}")
                
            except Exception as e:
                logger.error(f"Prediction failed: {str(e)}")
                return jsonify({
                    "error": "Model prediction failed",
                    "details": str(e)
                }), 500

            # Draw rectangle on original image
            draw = ImageDraw.Draw(img)
            draw.rectangle(
                [bbox['x'], bbox['y'], bbox['x'] + bbox['width'], bbox['y'] + bbox['height']],
                outline="red", width=3
            )

            # Re-encode processed images to base64
            try:
                buffered = io.BytesIO()
                cropped.save(buffered, format="PNG")
                processed_b64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
                processed_data_url = f"data:image/png;base64,{processed_b64}"
                
                logger.info(f"Image processed and encoded successfully")
                
            except Exception as e:
                logger.error(f"Failed to encode image: {str(e)}")
                return jsonify({
                    "error": "Failed to encode result image",
                    "details": str(e)
                }), 500
            
            logger.info(f"Results: Error-{error}, Comment-{label}, Confidence-{confidence}")
            
            return jsonify({
                "success": True,
                "processed_image": processed_data_url,
                "category": category,
                "view": view,
                "comment": label,
                "error": float(error),
                "confidence": float(confidence),
                "threshold": threshold,
                "diagnosis": diagnosis,
                "source": "pc"
            }), 200

        elif source == "telegram":
            # Handle Telegram source (form-data with file upload)
            logger.info("Processing Telegram source request")
            
            category = request.form.get('category')
            view = request.form.get('view')
            source_field = request.form.get('source')
            image_file = request.files.get('image')
            
            if not image_file:
                logger.error("No image file provided")
                return jsonify({
                    "error": "No image file provided",
                    "details": "Form must include 'image' file"
                }), 400
                
            if not category or not view:
                logger.error(f"Missing category or view: category={category}, view={view}")
                return jsonify({
                    "error": "Missing required fields",
                    "details": "category and view are required"
                }), 400

            try:
                img = Image.open(image_file)
                
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                logger.info(f"Image loaded successfully: {img.size}")
                    
            except Exception as e:
                logger.error(f"Invalid image file: {str(e)}")
                return jsonify({
                    "error": "Invalid image file",
                    "details": str(e)
                }), 400

            logger.info(f"Parameters:[Category-{category}, View-{view}, Image size-{img.size}]")
            logger.info("Processing image...")
            
            try:
                error = predict(img, view, category)
                label = get_label(error)
                confidence = 100 - (100 * float(error))
                diagnosis = build_diagnosis(error, confidence)
                
                logger.info(f"Prediction successful: Error-{error}, Confidence-{confidence}")
                
            except Exception as e:
                logger.error(f"Prediction failed: {str(e)}")
                return jsonify({
                    "error": "Model prediction failed",
                    "details": str(e)
                }), 500
            
            # Convert image back to base64 for response
            try:
                buffered = io.BytesIO()
                img.save(buffered, format="PNG")
                processed_b64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
                processed_data_url = f"data:image/png;base64,{processed_b64}"
                
                logger.info("Image encoded successfully")
                
            except Exception as e:
                logger.error(f"Failed to encode image: {str(e)}")
                return jsonify({
                    "error": "Failed to encode result image",
                    "details": str(e)
                }), 500

            response_data = {
                "success": True,
                "processed_image": processed_data_url,
                "category": category,
                "view": view,
                "comment": label,
                "error": float(error),
                "confidence": float(confidence),
                "threshold": threshold,
                "diagnosis": diagnosis,
                "source": "telegram"
            }

            logger.info(f"Results: Error-{error}, Comment-{label}, Confidence-{confidence}")
            
            return jsonify(response_data), 200
            
        else:
            logger.error(f"Unsupported source: {source}")
            return jsonify({
                "error": "Unsupported source",
                "details": f"Source must be 'pc' or 'telegram', got '{source}'"
            }), 400

    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            "error": "Internal server error",
            "details": str(e),
            "type": type(e).__name__
        }), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "service": "FACS API"
    }), 200

def get_label(error):
    """Generate label based on error threshold"""
    label = "⚠ Anomaly detected" if error > threshold else "✅ Normal structure"
    return label

def build_diagnosis(error, confidence):
    """
    Build diagnosis based on error and confidence scores.
    RUBRIC: Classification logic for diagnosis recommendations
    """
    diagnosis = ""
    
    if confidence < 100 and confidence > 98 and error > threshold:
        diagnosis = 'High probability that there is an anomaly in the structure.'
    elif confidence < 98 and confidence > 96 and error > threshold:
        diagnosis = 'My analysis concludes there could be an anomaly in the structure, but either the structure is not well annotated or the image could be distorted, small, unclear or bad hence confidence is reduced.'
    elif confidence < 96 and confidence > 92 and error > threshold:
        diagnosis = 'High possibility that anomaly is false positive and image is greatly distorted or irrelevant. Please check the image or the annotation and try again. If results are same, consider uploading a different scan.'
    elif confidence < 92 and error > threshold:
        diagnosis = 'Please upload a good ultrasound scan to obtain diagnosis. I cannot recognize the image nor the outlined structure.'
    elif confidence < 100 and confidence > 98 and error < threshold:
        diagnosis = 'Healthy structure detected. Annotation is correct OR model partially detects healthy area.'
    else:
        diagnosis = 'Analysis complete. Please review results with a medical professional.'
    
    diagnosis += " | ⚠️ THIS IS NOT PROFESSIONAL MEDICAL ADVICE. LIAISE WITH AN EXPERT"
    
    return diagnosis

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({
        "error": "Endpoint not found",
        "path": request.path
    }), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    logger.error(f"Internal server error: {str(error)}")
    return jsonify({
        "error": "Internal server error",
        "details": str(error)
    }), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
