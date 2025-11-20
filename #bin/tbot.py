import logging
import aiohttp
import os, time
from datetime import datetime
from io import BytesIO

from aiohttp import FormData
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application, CommandHandler, MessageHandler,
    filters, CallbackQueryHandler, ContextTypes
)

# PDF generation imports
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.platypus.tableofcontents import TableOfContents
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from PIL import Image as PILImage

from dotenv import load_dotenv
load_dotenv(dotenv_path='env.env')

#config
#TOKEN = "8320924107:AAH505mhHkOxeY3aLk0GObIpO_KCtY9hhLM"
TOKEN = os.getenv("TOKEN")
API_ENDPOINT = os.getenv("API_ENDPOINT")
logging.basicConfig(level=logging.INFO)

#category mappings
VIEW_CATEGORIES = {
    "crl": {
        "Maxilla": "mx",
        "Mandible-MDS": "mds", 
        "Mandible-MLS": "mls",
        "Lateral ventricle": "lv",
        "Head": "head",
        "Gestational sac": "gsac",
        "Thorax": "thorax",
        "Abdomen": "ab",
        "Body(Biparietal diameter)": "bd",
        "Rhombencephalon": "rbp",
        "Diencephalon": "dp",
        "NTAPS": "ntaps",
        "Nasal bone": "nb"
    },
    "nt": {
        "Maxilla": "mx",
        "Mandible-MDS": "mds", 
        "Mandible-MLS": "mls",
        "Lateral ventricle": "lv",
        "Head": "head",
        "Thorax": "thorax",
        "Abdomen": "ab",
        "Rhombencephalon": "rbp",
        "Diencephalon": "dp",
        "Nuchal translucency": "nt",
        "NTAPS": "ntaps",
        "Nasal bone": "nb"
    }
}

def generate_medical_report_pdf(image_path, result_data, view, category, user_id):
    """Generate a professional medical report PDF"""
    
    # Create filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pdf_filename = f"Analysis-report_{user_id}_{timestamp}.pdf"
    
    # Create the PDF document
    doc = SimpleDocTemplate(pdf_filename, pagesize=A4,
                          topMargin=0.8*inch, bottomMargin=0.8*inch,
                          leftMargin=0.8*inch, rightMargin=0.8*inch)
    
    # Container for the 'Flowable' objects
    story = []
    
    # Define custom styles with left alignment
    styles = getSampleStyleSheet()
    
    # Hospital header style - LEFT ALIGNED
    header_style = ParagraphStyle(
        'CustomHeader',
        parent=styles['Heading1'],
        fontSize=20,
        textColor=colors.HexColor('#1f4e79'),  # Medical blue
        spaceAfter=6,
        alignment=TA_LEFT,  # Changed to LEFT
        fontName='Helvetica-Bold'
    )
    
    # Subheader style - LEFT ALIGNED
    subheader_style = ParagraphStyle(
        'CustomSubHeader',
        parent=styles['Heading2'],
        fontSize=12,
        textColor=colors.HexColor('#2980b9'),
        spaceAfter=12,
        alignment=TA_LEFT,  # Changed to LEFT
        fontName='Helvetica-Bold'
    )
    
    # Section header style
    section_style = ParagraphStyle(
        'SectionHeader',
        parent=styles['Heading3'],
        fontSize=10,
        textColor=colors.HexColor('#1f4e79'),
        spaceAfter=6,
        spaceBefore=12,
        fontName='Helvetica-Bold'
    )
    
    # Body text style
    body_style = ParagraphStyle(
        'BodyText',
        parent=styles['Normal'],
        fontSize=10,
        textColor=colors.black,
        spaceAfter=6,
        fontName='Helvetica'
    )
    
    # Add header with icon
    try:
        # Create a table for the main header with icon
        if os.path.exists('ico.png'):
            icon_img = Image('ico.png', width=30, height=30)
            header_data = [[icon_img, "FACS MEDICAL IMAGING REPORT"]]
            header_table = Table(header_data, colWidths=[0.4*inch, 5.6*inch])
            header_table.setStyle(TableStyle([
                ('ALIGN', (0, 0), (0, 0), 'LEFT'),
                ('ALIGN', (1, 0), (1, 0), 'LEFT'),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('FONTNAME', (1, 0), (1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (1, 0), (1, 0), 21),
                ('TEXTCOLOR', (1, 0), (1, 0), colors.HexColor('#1f4e79')),
                ('LEFTPADDING', (0, 0), (-1, -1), 0),
                ('RIGHTPADDING', (0, 0), (-1, -1), 0),
                ('TOPPADDING', (0, 0), (-1, -1), 0),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 0),
            ]))
            story.append(header_table)
        else:
            story.append(Paragraph("FACS MEDICAL REPORT(FMR)", header_style))
    except Exception as e:
        logging.error(f"Error loading header icon: {e}")
        story.append(Paragraph("FACS MEDICAL REPORT(FMR)", header_style))
    
    story.append(Paragraph("Ultrasound Analysis", subheader_style))
    
    # Report information table
    current_time = datetime.now().strftime("%B %d, %Y at %I:%M %p")
    report_data = [
        ['Report Date:', current_time],
        ['Patient ID:', f'TG_{user_id}'],
        ['Report ID:', f'RPT_{timestamp}'],
        ['Analysis Type:', 'FPC - Fetal plane classification']
    ]
    
    report_table = Table(report_data, colWidths=[2*inch, 3*inch])
    report_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#f8f9fa')),
        ('TEXTCOLOR', (0, 0), (0, -1), colors.HexColor('#1f4e79')),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#dee2e6')),
        ('ROWBACKGROUNDS', (0, 0), (-1, -1), [colors.white, colors.HexColor('#f8f9fa')])
    ]))
    
    story.append(report_table)
    
    # Clinical Information Section
    story.append(Paragraph("CLINICAL INFORMATION", section_style))
    
    clinical_data = [
        ['Ultrasound View:', view.upper()],
        ['Anatomical Category:', category],
        ['Analysis Method:', 'Feature extraction']
    ]
    
    clinical_table = Table(clinical_data, colWidths=[2*inch, 3*inch])
    clinical_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#e3f2fd')),
        ('TEXTCOLOR', (0, 0), (0, -1), colors.HexColor('#1565c0')),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#bbdefb')),
    ]))
    
    story.append(clinical_table)
    
    # Ultrasound Image Section
    story.append(Paragraph("ULTRASOUND IMAGE", section_style))
    
    # Add the ultrasound image
    if os.path.exists(image_path):
        try:
            # Open and resize image if necessary
            pil_img = PILImage.open(image_path)
            img_width, img_height = pil_img.size
            
            # Calculate scaling to fit within bounds
            max_width = 4*inch
            max_height = 3*inch
            
            if img_width > max_width or img_height > max_height:
                ratio = min(max_width/img_width, max_height/img_height)
                new_width = img_width * ratio
                new_height = img_height * ratio
            else:
                new_width = img_width
                new_height = img_height
            
            # Add image to PDF
            img = Image(image_path, width=new_width, height=new_height)
            story.append(img)
            
        except Exception as e:
            story.append(Paragraph(f"Image could not be loaded: {str(e)}", body_style))
    
    # Analysis Results Section
    story.append(Paragraph("ANALYSIS RESULTS", section_style))
    
    # Results table
    confidence = result_data.get('confidence', 0)
    error = result_data.get('error', 0)
    status = result_data.get('comment', 'No comment')
    diagnosis = result_data.get('diagnosis', 'No diagnosis available')
    
    # Color coding for confidence levels
    if confidence >= 80:
        confidence_color = colors.HexColor('#d4edda')  # Light green
    elif confidence >= 60:
        confidence_color = colors.HexColor('#fff3cd')  # Light yellow
    else:
        confidence_color = colors.HexColor('#f8d7da')  # Light red
    
    results_data = [
        ['Parameter', 'Value'],
        ['Confidence Level', f'{confidence:.2f}%'],
        ['Reconstruction Error', f'{error:.5f}',],
        ['Analysis Status', status,],
    ]
    
    results_table = Table(results_data, colWidths=[2*inch, 1.5*inch, 1.5*inch])
    results_table.setStyle(TableStyle([
        # Header row
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f4e79')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 11),
        ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
        
        # Data rows
        ('FONTNAME', (0, 1), (0, -1), 'Helvetica-Bold'),
        ('FONTNAME', (1, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 9),
        ('ALIGN', (0, 1), (0, -1), 'LEFT'),
        ('ALIGN', (1, 1), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        
        # Confidence row coloring
        ('BACKGROUND', (1, 1), (1, 1), confidence_color),
        
        # Grid and borders
        ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#dee2e6')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f8f9fa')])
    ]))
    
    story.append(results_table)
    
    # Diagnosis Section with icon
    story.append(Paragraph("DIAGNOSIS", section_style))
    
    # Diagnosis box with proper text wrapping and left alignment
    diagnosis_lines = diagnosis.split('\n')
    formatted_diagnosis = '<br/>'.join([line.strip() for line in diagnosis_lines if line.strip()])
    
    diagnosis_paragraph = Paragraph(formatted_diagnosis, ParagraphStyle(
        'DiagnosisText',
        parent=body_style,
        fontSize=9,
        textColor=colors.HexColor('#2e7d32'),
        leftIndent=12,
        rightIndent=12,
        spaceAfter=0,
        spaceBefore=0,
        alignment=TA_LEFT,
        fontName='Helvetica'
    ))
    
    diagnosis_data = [[diagnosis_paragraph]]
    diagnosis_table = Table(diagnosis_data, colWidths=[5*inch])
    diagnosis_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#e8f5e8')),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('BOX', (0, 0), (-1, -1), 2, colors.HexColor('#4caf50')),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
        ('TOPPADDING', (0, 0), (-1, -1), 12),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
    ]))
    
    story.append(diagnosis_table)
    
    # Footer
    story.append(Spacer(1, 0.3*inch))
    footer_style = ParagraphStyle(
        'Footer',
        parent=styles['Normal'],
        fontSize=4,
        textColor=colors.HexColor('#666666'),
        alignment=TA_CENTER,
        fontName='Helvetica'
    )
    
    # Build the PDF
    doc.build(story)
    
    return pdf_filename

#start
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Send an ultrasound image (JPG/png only) to begin. See /instructions first")

async def instructions(update: Update, context: ContextTypes.DEFAULT_TYPE):
    instruction_message = """
📋 Instructions

🔸 Step 1: Send a cropped ultrasound image of a structure you want to analyze. See all structures with /list command
🔸 Step 2: Select the ultrasound view (CRL or NT)
🔸 Step 3: Choose the anatomical category to analyze
🔸 Step 4: Wait for the AI analysis results
🔸 Step 5: Download the generated PDF report

📌 Important notes:
• Only JPG/PNG images are supported
• Ensure the ultrasound image is clear and properly oriented
• Results are for reference only - always consult a medical professional
• Processing may take a few seconds
• A PDF report will be automatically generated after analysis
• Stop bot with /stop

💡 Tips:
• Use high-quality, well-lit images for better accuracy
• Make sure the anatomical structure is clearly visible
• Different views (CRL/NT) have different category options

🆘 Need help? Contact support if you encounter any issues @d3ikshr.
    """
    
    await update.message.reply_text(instruction_message, parse_mode='HTML')

async def list_categories(update: Update, context: ContextTypes.DEFAULT_TYPE):
    list_message = """
📋 Available Categories by View

🔍 CRL view categories:
• Maxilla • Mandible-MDS • Mandible-MLS
• Lateral ventricle • Head • Gestational sac
• Thorax • Abdomen • Body(Biparietal diameter)
• Rhombencephalon • Diencephalon • NTAPS
• Nasal bone

🔍 NT view categories:
• Maxilla • Mandible-MDS • Mandible-MLS
• Lateral ventricle • Head • Thorax
• Abdomen • Rhombencephalon • Diencephalon
• Nuchal translucency • NTAPS • Nasal bone

💡 Note: Categories will be shown automatically based on your selected view during analysis.
    """
    
    await update.message.reply_text(list_message, parse_mode='HTML')

async def stop(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("🛑 Bot stopped for this chat. Use /start to begin again.")
    # Clear user data
    context.user_data.clear()

# Receive image
async def handle_image(update: Update, context: ContextTypes.DEFAULT_TYPE):
    photo = update.message.photo[-1]
    file = await photo.get_file()
    file_path = f"{update.message.from_user.id}_ultrasound.jpg"
    await file.download_to_drive(file_path)
    context.user_data["image_path"] = file_path
    
    # get view
    buttons = [
        [InlineKeyboardButton("CRL", callback_data="view:crl"),
         InlineKeyboardButton("NT", callback_data="view:nt")]
    ]
    await update.message.reply_text(
        "Select the ultrasound view:",
        reply_markup=InlineKeyboardMarkup(buttons)
    )

#selcet view
async def handle_view(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    view = query.data.split(":")[1]
    context.user_data["selected_view"] = view
    
    #get categories for the selected view
    categories = list(VIEW_CATEGORIES[view].keys())
    buttons = [[InlineKeyboardButton(cat, callback_data=f"category:{cat}")]
               for cat in categories]
    await query.edit_message_text(
        "Select anatomical category:",
        reply_markup=InlineKeyboardMarkup(buttons)
    )

#get category then upload
async def handle_category(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    category_display = query.data.split(":")[1]
    context.user_data["selected_category"] = category_display
    
    image_path = context.user_data.get("image_path")
    view = context.user_data.get("selected_view")
    user_id = query.from_user.id
    
    if not image_path or not view:
        await query.edit_message_text("Missing image or view.")
        return

    await query.edit_message_text("🔄 Processing image...")
    time.sleep(3)
    await query.edit_message_text("🏥 Building diagnosis, please wait...")

    try:
        #read the image file into memory first
        with open(image_path, "rb") as f:
            image_data = f.read()
        
        #mapping
        category_value = VIEW_CATEGORIES[view][category_display]
        
        #create form data
        form = FormData()
        form.add_field("view", view)
        form.add_field("category", category_value)
        form.add_field("source", "telegram")
        form.add_field("image", image_data, filename="image.jpg", content_type="image/jpeg")
        
        #send request 
        async with aiohttp.ClientSession() as session:
            async with session.post(API_ENDPOINT, data=form) as resp:
                if resp.status == 200:
                    result = await resp.json()
                    
                    message = f"""
📊 Analysis Results

🔍 View: {result.get('view', view).upper()}
🏥 Category: {category_display}

📈 Confidence: {result.get('confidence', 0):.2f}%
⚠️ Reconstruction error: {result.get('error', 0):.5f}

📋 Status: {result.get('comment', 'No comment')}

🩺 Diagnosis: {result.get('diagnosis', 'No diagnosis')}

📄 Generating PDF report...
                    """
                    
                    await query.edit_message_text(message, parse_mode='HTML')
                    
                    # Generate PDF report
                    try:
                        pdf_filename = generate_medical_report_pdf(
                            image_path, result, view, category_display, user_id
                        )
                        
                        # Send the PDF file
                        with open(pdf_filename, 'rb') as pdf_file:
                            await query.message.reply_document(
                                document=pdf_file,
                                filename=f"Analysis-report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                                caption="📄Report is ready!\nFor reference only."
                            )
                        
                        # Clean up PDF file
                        if os.path.exists(pdf_filename):
                            os.remove(pdf_filename)
                            
                        logging.info(f"PDF report generated and sent for user {user_id}")
                        
                    except Exception as pdf_error:
                        logging.error(f"Error generating PDF: {pdf_error}")
                        await query.message.reply_text("❌ Could not generate PDF report, but analysis is complete.")
                     
                else:
                    error_text = await resp.text()
                    await query.edit_message_text(f"❌ Upload failed. Status: {resp.status}\nError: {error_text}")
                    
    except Exception as e:
        logging.error(f"Error processing request: {e}")
        await query.edit_message_text(f"❌ Error: {str(e)}")
        
    finally:
        try:
            if os.path.exists(image_path):
                os.remove(image_path)
                logging.info(f"Cleaned up image file: {image_path}")
        except Exception as e:
            logging.error(f"Error cleaning up file: {e}")

def main():
    app = Application.builder().token(TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("instructions", instructions))
    app.add_handler(CommandHandler("list", list_categories))
    app.add_handler(CommandHandler("stop", stop))
    app.add_handler(MessageHandler(filters.PHOTO, handle_image))
    app.add_handler(CallbackQueryHandler(handle_view, pattern="^view:"))
    app.add_handler(CallbackQueryHandler(handle_category, pattern="^category:"))
    app.run_polling()

if __name__ == "__main__":
    main()