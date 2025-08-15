import fitz  # PyMuPDF library for PDF handling
import json

def extract_text_from_pdf(pdf_path):
    # Open the PDF file
    pdf_document = fitz.open(pdf_path)
    text_data = []

    # Loop through each page in the PDF
    for page_num in range(pdf_document.page_count):
        page = pdf_document[page_num]
        page_text = page.get_text("text")  # Extract text from the page
        text_data.append({"page_num": page_num + 1, "content": page_text})

    pdf_document.close()
    return text_data


# Path to your PDF file
pdf_path = "/Users/mathewyoussef/Desktop/Windward_Bound/Windward_Bound_Information.pdf"
text_data = extract_text_from_pdf(pdf_path)

# Verify output by printing the text from each page
for page in text_data:
    print(f"Page {page['page_num']}:\n{page['content']}\n{'='*40}")

# Save the extracted text to a JSON file
with open("extracted_text.json", "w") as json_file:
    json.dump(text_data, json_file, indent=4)

print("PDF text successfully saved to extracted_text.json")