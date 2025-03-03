from openai import OpenAI
import base64
import pymupdf
from os import path
import re

class GPTProcessor:
    def __init__(self, api_key):
        self.client = OpenAI(api_key=api_key)
        self.model = "gpt-4o-mini"

        self.base_dir = "processed"
        self.images_path = path.join(self.base_dir, "images")


    # Get each pages as a base64 encoded images
    def get_pages(self, file_path):
        print("Starting page extraction...")
        doc = pymupdf.open(file_path)
        base64_pages = []

        for page in (doc):
            pix = page.get_pixmap()  # render page to an image
            img_bytes = pix.tobytes()

            img_base64 = base64.b64encode(img_bytes).decode("utf-8")

            base64_pages.append(img_base64)
        print("Page extraction complete.")
        return base64_pages
    

    def get_alt_text(self, pages):
        print("Starting alt text generation...")

        SYSTEM_PROMPT = """
        You are a helpful assistant that generates highly detailed and descriptive alt text for images and figures.
        You will be given a list of images that represent sequential pages of a document.
        Your task is to generate a detailed alt text description for each image in those pages.
        
        Requirements:
        1. You output should consist of a list of key-value pairs, where the key is the image name and the value is the alt text description.
        2. The alt text description should be preceded by the words: "Alt text:".
        3. Even if a figure or image has a caption, still generate an alt text description for it.
        4. Do not include any image links or filepaths in the alt text.
        5. Return no other text than the list of key-value pairs.
        """

        USER_PROMPT = """
        Generate highly detailed and descriptive alt text for all the images in this page.
        
        Return a list of key-value pairs, where the key is the image name and the value is the alt text description.
        The alt text description should be preceded by the words: "Alt text:".
        Return no other text than the list of key-value pairs.
        """

        try:
            alt_text = []
            for i, page in enumerate(pages):
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": [{"type": "text", "text": USER_PROMPT}, {
                            "type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{page}"}}]}
                    ]
                )
                print(f"Finished alt text for page {i+1}")
                alt_text.append(response.choices[0].message.content)
        except Exception as e:
            print(f"Error generating transcript: {e}")
            return None
        return alt_text
    
    def get_structured_md_incremental(self, pages):
        print("Starting incremental processing...")
        structured_pages = []
        
        # Get alt text and transcription for each page
        # alt_texts = self.get_alt_text(pages)
        # if not alt_texts:
        #     return None
            
        # page_transcripts = self.get_raw_transcription(pages)
        # if not page_transcripts:
        #     return None

        # Process each page individually
        for i, page in enumerate(pages):
            print(f"Structuring page {i+1}...")

            page_alt_text = self.get_alt_text([page])
            page_transcript = self.get_raw_transcription([page])
            
            SYSTEM_PROMPT = """
            Given a markdown transcript and a list of alt text descriptions for images, change heading and subheading levels as needed and inject the alt text descriptions into the right places to return a properly formatted, logically structured markdown document.
            
            Requirements:
            1. Convert any actual references to actual images, including links, brackets, filepaths, or source information in the markdown to plain markdown.
            2. Leave all original text and captions intact and in the same place they were in the original markdown.
            3. Remove any and all markdown code blocks.
            4. Return no other text than the markdown.
            """

            USER_PROMPT = f"""
            Given a markdown transcript and a list of alt text descriptions for images, reformat headings and subheadings as needed and inject the alt text descriptions into the right places to return a properly formatted, logically structured markdown document.
            
            Here is the list of alt text descriptions for images:
            {page_alt_text}

            Here is the markdown transcript:
            {page_transcript}
            """

            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": USER_PROMPT}
                    ]
                )
                structured_page = response.choices[0].message.content.replace("```md", "").replace("```", "").replace("markdown", "")
                structured_page = re.sub(r"!\[(.*?)\]", r"\1", structured_page)

                structured_pages.append(structured_page)
            except Exception as e:
                print(f"Error processing page {i+1}: {e}")
                return None

        # Combine all pages into a single markdown document
        print("Combining pages...")
        combined_markdown = "\n\n".join(structured_pages)
        
        return combined_markdown
        

    def get_images(self, file_path):
        print("Starting image extraction...")
        doc = pymupdf.open(file_path)

        for page in (doc):
            image_list = page.get_images()
            for image in image_list:
                xref = image[0]
                pix = pymupdf.Pixmap(doc, xref)
                if pix.colorspace and pix.colorspace.name not in ["RGB", "L"]:
                    pix = pymupdf.Pixmap(pymupdf.csRGB, pix)
                pix.save(f"{self.images_path}/page_{page.number}-{xref}.png")

                img_bytes = pix.tobytes()
                img_base64 = base64.b64encode(img_bytes).decode("utf-8")
                self.base64_images.append(img_base64)

    def get_raw_transcription(self, pages):
        print("Starting transcript generation...")

        SYSTEM_PROMPT = """
        You are a helpful assistant that can accurately and precisely extract text from images.
        You will be given a list of images that represent sequential pages of a document.
        Your task is to extract the text from each page and return it in a structured markdown format.
 

        Requirements:
        1. Maintain logical reading order. If the text is in columns, transcribe the text from top to bottom, left to right in the columns. The columns may be broken by images.
        2. Maintain all image captions and references.
        3. Maintain proper headings, subheadings, and hierarchies.
        4. Do not include any footers, headers, page numbers, or other metadata.
        5. For any images in the page, generate a detailed alt text description for someone who is blind or low vision and include it directly following the caption.
        6. Do not include any actual image links or filepaths in the transcript.
        7. Return no other text than the transcript.

        """

        USER_PROMPT = """
        Transcribe this page of a document, maintaining proper headings, subheadings, and hierarchies.
        Return no other text than the transcript.
        """

        try:
            transcript = ""
            for i, page in enumerate(pages):
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": [{"type": "text", "text": USER_PROMPT}, {
                            "type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{page}"}}]}
                    ]
                )
                print(f"Finished transcribing page {i+1}")
                transcript += response.choices[0].message.content
        except Exception as e:
            print(f"Error generating raw transcription: {e}")
            return None
        return transcript
    
    def get_structured_md(self, raw_transcription, alt_text):
        print("Starting transcript structuring...")

        SYSTEM_PROMPT = """
        Given a markdown transcript and a list of alt text descriptions for images, change heading and subheading levels as needed and inject the alt text descriptions into the right places to return a properly formatted, logically structured markdown document.
        
        Requirements:
        1. Remove any actual image links, filepaths, or source information in the markdown.
        2. Leave all original text and captions intact and in the same place they were in the original markdown.
        3. Return no other text than the markdown.
        """

        USER_PROMPT = f"""
        Given a markdown transcript and a list of alt text descriptions for images, reformat headings and subheadings as needed and inject the alt text descriptions into the right places to return a properly formatted, logically structured markdown document.
        
        Here is the list of alt text descriptions for images:
        {alt_text}

        Here is the markdown transcript:
        {raw_transcription}
        """

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": [{"type": "text", "text": USER_PROMPT}]}
                ]
            )
            structured_transcript = response.choices[0].message.content.replace("```md", "").replace("```", "")
        except Exception as e:
            print(f"Error generating structured transcription  : {e}")
            return None
        return structured_transcript

    def get_structured_html(self, raw_transcription, alt_text):
        print("Starting transcript structuring...")

        SYSTEM_PROMPT = """
        You are a helpful assistant that can accurately and precisely convert and restructure markdown into properly formatted HTML.
        Given a a markdown transcript and a list of alt text descriptions for images, inject the alt text descriptions into the right places and return a properly formatted HTML document that maintains the original heading, hierarchy, and captions.
        Remove any actual image links, filepaths, or source information in the HTML.
        Return no other text than the HTML.
        """

        USER_PROMPT = f"""
        Convert the following markdown into HTML, adjusting headings, subheadings, and hierarchies as necessary to main proper hierarchy. Maintain image captions and references.
        
        Here is the list of alt text descriptions for images:
        {alt_text}

        Here is the markdown transcript:
        {raw_transcription}
        """

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": [{"type": "text", "text": USER_PROMPT}]}
                ]
            )
            structured_transcript = response.choices[0].message.content.replace("```html", "").replace("```", "")
        except Exception as e:
            print(f"Error generating structured transcription  : {e}")
            return None
        return structured_transcript
    
    def convert_md_to_html(self, md_text):
        print("Starting markdown to HTML conversion...")

        SYSTEM_PROMPT = """
        You are a helpful assistant that can accurately and precisely convert and restructure markdown into properly formatted HTML.
        """

        USER_PROMPT = f"""
        Convert the following markdown into a clean, well-structured HTML document. Maintain all original text. Alt text descriptions should be preceded by the words: "Alt text:".
        
        Here is the markdown transcript:
        {md_text}
        """

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": [{"type": "text", "text": USER_PROMPT}]}
                ]
            )
            html_text = response.choices[0].message.content.replace("```html", "").replace("```", "")
        except Exception as e:
            print(f"Error converting markdown to HTML: {e}")
            return None
        return html_text