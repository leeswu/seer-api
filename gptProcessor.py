from openai import OpenAI
import base64
import pymupdf
from os import path


class GPTProcessor:
    def __init__(self, api_key):
        self.client = OpenAI(api_key=api_key)
        self.base64_pages = []
        self.base64_images = []

        self.base_dir = "processed"
        self.images_path = path.join(self.base_dir, "images")

    def get_pages(self, file_path):
        print("Starting page extraction...")
        doc = pymupdf.open(file_path)

        for page in (doc):
            pix = page.get_pixmap()  # render page to an image
            img_bytes = pix.tobytes()

            img_base64 = base64.b64encode(img_bytes).decode("utf-8")

            self.base64_pages.append(img_base64)
        print("Page extraction complete.")

    def get_alt_text(self):
        print("Starting transcript generation...")

        SYSTEM_PROMPT = """
        You are a helpful assistant that generates highly detailed and descriptive alt text for images.
        You will be given a list of images that represent sequential pages of a document.
        Your task is to generate a detailed alt text description for each image in those pages.
        
        You output should consist of a list of key-value pairs, where the key is the image name and the value is the alt text description.
        Do not include any other text than the list of key-value pairs.
        Do not include any image links or filepaths in the alt text.
        """

        USER_PROMPT = """
        Generate highly detailed and descriptive alt text for all the images in this page.
        
        Return a list of key-value pairs, where the key is the image name and the value is the alt text description.
        Do not include any other text than the list of key-value pairs.
        """

        if len(self.base64_pages) == 0:
            print("No images found.")
            return None

        try:
            alt_text = []
            for i, page in enumerate(self.base64_pages):
                response = self.client.chat.completions.create(
                    model="gpt-4o-mini",
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

    def get_raw_transcript(self):
        print("Starting transcript generation...")

        SYSTEM_PROMPT = """
        You are a helpful assistant that can accurately and precisely extract text from images.
        You will be given a list of images that represent sequential pages of a document.
        Your task is to extract the text from each page and return it in a structured markdown format.
 

        Requirements:
        1. Maintain all image captions and references.
        2. Maintain proper headings, subheadings, and hierarchies.
        3. Do not include any footers, headers, page numbers, or other metadata.
        4. For any images in the page, generate a detailed alt text description for someone who is blind or low vision and include it directly following the caption.
        5. Do not include any actual image links or filepaths in the transcript.
        6. Return no other text than the transcript.

        """

        USER_PROMPT = """
        Transcribe this page of a document, maintaining proper headings, subheadings, and hierarchies.
        Return no other text than the transcript.
        """

        if len(self.base64_pages) == 0:
            print("No images found.")
            return None

        try:
            transcript = ""
            for i, page in enumerate(self.base64_pages):
                response = self.client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": [{"type": "text", "text": USER_PROMPT}, {
                            "type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{page}"}}]}
                    ]
                )
                print(f"Finished transcribing page {i+1}")
                transcript += response.choices[0].message.content
        except Exception as e:
            print(f"Error generating transcript: {e}")
            return None
        return transcript

    def get_structured_transcript(self, raw_transcript, alt_text):
        print("Starting transcript structuring...")

        SYSTEM_PROMPT = """
        You are a helpful assistant that can accurately and precisely convert and restructure markdown into properly formatted HTML.
        Given a a markdown transcript and a list of alt text descriptions for images, inject the alt text descriptions into the right places and return a properly formatted HTML document that maintains the original heading, hierarchy, and captions.
        Do not include any image links, filepaths, or source information in the HTML.
        Return no other text than the HTML.
        """

        USER_PROMPT = f"""
        Convert the following markdown into HTML, adjusting headings, subheadings, and hierarchies as necessary to main proper hierarchy. Maintain image captions and references.
        
        Here is the list of alt text descriptions for images:
        {alt_text}

        Here is the markdown transcript:
        {raw_transcript}
        """

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": [{"type": "text", "text": USER_PROMPT}]}
                ]
            )
            structured_transcript = response.choices[0].message.content.replace("```html", "").replace("```", "")
        except Exception as e:
            print(f"Error generating transcript: {e}")
            return None
        return structured_transcript

    def ask_about_content(self, question):
        # Assuming you have stored the content somewhere accessible
        # You might want to store it in a session or pass it along
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant answering questions about a PDF document's content."},
                {"role": "user", "content": question}
            ]
        )
        return response.choices[0].message.content
