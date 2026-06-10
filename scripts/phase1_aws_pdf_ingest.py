#Composer coding assistant through a Cursor IDE guided the development of this script

#script to run aws textract job in order to convert pdfs from DOE/NHS into machine-readable text files

import os
import boto3
import time
from dotenv import load_dotenv

load_dotenv()

#set AWS region and bucket name ready for textract job
S3_BUCKET_NAME = os.environ.get("S3_BUCKET_NAME", "sen-rag-dissertation")
AWS_REGION = (
    os.environ.get("AWS_REGION")
    or os.environ.get("AWS_DEFAULT_REGION")
    or "us-west-2"
)
INPUT_DIR = "../data/01_raw_pdfs/"
OUTPUT_DIR = "../data/02_extracted_text/"

#sets the client for uploading pdfs to s3 bucket and textract job
s3 = boto3.client("s3", region_name=AWS_REGION)
textract = boto3.client("textract", region_name=AWS_REGION)

#function to upload pdfs to s3 bucket
def upload_to_s3(file_path, bucket_name, object_name):
    """Uploads a file to an AWS S3 bucket."""
    print(f"Uploading {object_name} to S3...")
    s3.upload_file(file_path, bucket_name, object_name)
    return object_name

#function to start textract job for multi-page PDFs
def start_textract_job(bucket_name, document_key):
    """Starts an asynchronous Textract job for multi-page PDFs."""
    print(f"Starting Textract job for {document_key}...")
    response = textract.start_document_text_detection(
        DocumentLocation={'S3Object': {'Bucket': bucket_name, 'Name': document_key}}
    )
    return response['JobId']


#checks in on textract job every 5 seconds until it is complete
def get_textract_results(job_id):
    """Polls AWS until the Textract job is complete, then retrieves the text."""
    print(f"Waiting for Textract job to complete", end="")
    
    while True:
        response = textract.get_document_text_detection(JobId=job_id)
        status = response['JobStatus']
        if status == 'SUCCEEDED':
            break
        elif status == 'FAILED':
            raise Exception(f"Textract job failed: {job_id}")
        time.sleep(5) 
        print(".", end="", flush=True)
    
    print("\nJob complete. Extracting text...")
    
    # collects textract results pages and returns full doc as plain text
    pages = [response]
    while 'NextToken' in response:
        response = textract.get_document_text_detection(JobId=job_id, NextToken=response['NextToken'])
        pages.append(response)
        
    extracted_text = ""
    for page in pages:
        for item in page['Blocks']:
            if item['BlockType'] == 'LINE':
                extracted_text += item['Text'] + "\n"
                
    return extracted_text

#main function to loop through PDFs and process them
def main():
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

#sorts pdf files in input directory and processes them
    pdf_files = sorted(f for f in os.listdir(INPUT_DIR) if f.endswith(".pdf"))
    if not pdf_files:
        print(f"No PDFs found in {INPUT_DIR}")
        return
    

    print(f"Bucket: {S3_BUCKET_NAME} | Region: {AWS_REGION} | PDFs: {len(pdf_files)}\n")

    # Loop through PDFs
    for filename in pdf_files:
        file_path = os.path.join(INPUT_DIR, filename)
        object_name = filename
        output_filename = filename.replace(".pdf", ".txt")
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        if os.path.isfile(output_path):
            print(f"Skip (exists): {output_filename}\n")
            print("-" * 40)
            continue

        # 1. Upload to S3
        upload_to_s3(file_path, S3_BUCKET_NAME, object_name)

        # 2. Start Textract
        job_id = start_textract_job(S3_BUCKET_NAME, object_name)

        # 3. Get Results
        extracted_text = get_textract_results(job_id)

        # 4. Save locally to .txt file
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(extracted_text)
#prints success message and line break with 40 - characters for readability
        print(f"extracted and saved: {output_filename}\n")
        print("-" * 40)

#main function to run the script and alert when job is in the folder
if __name__ == "__main__":
    print("Starting Phase 1: AWS Textract PDF -> text extraction\n")
    main()
    print("Finished processing PDFs and saved to data/02_extracted_text/")

    