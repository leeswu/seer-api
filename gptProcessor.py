from openai import OpenAI
import base64
import pymupdf
from os import path
import re
import json


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
        Base your output on the requirements and the example below.

        Requirements:
        1. You output should consist of a list of new line separated JSON objects, structured as follows: {"imageName": <IMAGE_NAME>, "altText": <ALT_TEXT>}
        where the name of the image or figure should go where <IMAGE_NAME> is and the alt text should go where <ALT_TEXT> is.
        2. Even if a figure or image has a caption, still generate an alt text description for it. Do not generate more than one alt text per image or figure.
        3. Do not include any image links or filepaths in the alt text.
        4. If a figure or image is unnamed, replace the <IMAGE_NAME> with "Unlabeled Figure" followed by a short 3-5 word descriptive name for the figure.
        5. Return no other text than the list of JSON objects.

        Example:
        {"imageName": "Figure 1.", "altText": "This is the alt text for figure 1"}\n
        {"imageName": "Figure 8.", "altText": "This is some alt text for figure 8"}\n
        {"imageName": "Unlabeled Figure.",
            "altText": "This is alt text for an unlabeled figure."}
        """

        USER_PROMPT = """
        Generate highly detailed and descriptive alt text for all the images in this page.

        Return a list of new line separated JSON objects that contain alt text for each image, structured as follows: {"imageName": <IMAGE_NAME>, "altText": <ALT_TEXT>}
        where the name of the image or figure should go where <IMAGE_NAME> is and the alt text should go where <ALT_TEXT> is.
        Return no other text than the JSON objects.
        """

        try:
            alt_texts = []

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

                alt_text = response.choices[0].message.content.split("\n")
                print("ALT TEXT AFTER SPLIT:")
                print(alt_text)
                alt_text = [json.loads(alt)
                            for alt in alt_text if self._is_json(alt)]

                for alt in alt_text:
                    alt["altText"] = "AI-Generated Alt Text: " + alt["altText"]
                alt_texts.extend(alt_text)

        except Exception as e:
            print(f"Error generating transcript: {e}")
            return None
        return alt_texts

    def _is_json(self, myjson):
        try:
            json.loads(myjson)
        except ValueError as e:
            return False
        return True

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
            You are a helpful assistant that carefully and properly reconstructs markdown transcripts.

            Given a markdown transcript and a list of alt text descriptions for images in the form of JSON objects, reconstruct the the raw markdown
            transcript by injecting the alt text into the appropriate places. Ensure that you output a properly formatted, logically structured markdown document.

            Base your output on the requirements and example input and example output below.

            Requirements:
            1. Remove any and all references to actual image content, including links, brackets, filepaths, or source information in the markdown to plain markdown.
            2. Leave all original text and captions intact and in the same place they were in the raw transcript.
            3. Inject alt text for the image immediately after the existing image caption.
            4. Remove any and all markdown code blocks.
            5. Return no other text than the markdown.

            Example Input:

            Alt Text: [{"figureName": "Figure 1", "altText": "AI-Generated Alt Text: This is alt text for figure 1."}]

            Raw Transcript:

            # Main Title of the Document

            Here is some main body text.

            ### Figure 1. A caption for figure 1.

            Example Output:

            # Main Title of the Document

            Here is some main body text.

            ### Figure 1. A caption for figure 1.
            AI-Generated Alt Text: This is alt text for figure 1.
            """

            USER_PROMPT = f"""
            Given a markdown transcript and a list of alt text descriptions for images in the form of JSON objects, reconstruct the the raw markdown
            transcript by injecting the alt text into the appropriate places. Ensure that you output a properly formatted, logically structured markdown document.

            Alt Text:
            {page_alt_text}

            Raw Transcript:
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
                structured_page = response.choices[0].message.content.replace(
                    "```md", "").replace("```", "").replace("markdown", "")
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
        3. Enforce proper headings, subheadings, and hierarchies. If two pieces of text are the same size, they should have the same heading level.
        4. Do not include any footers, headers, page numbers, or other metadata.
        5. For any images in the page, generate a detailed alt text description for someone who is blind or low vision and include it directly following the caption.
        6. Do not include any actual image links or filepaths in the transcript.
        7. Return no other text than the transcript.

        """

        USER_PROMPT = """
        Transcribe this document page, maintaining proper headings, subheadings, and hierarchies.
        Return no other text than the transcript.
        """

        try:
            transcript = ""
            prev_message = []

            for i, page in enumerate(pages):
                message = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": [{"type": "text", "text": USER_PROMPT}, {
                        "type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{page}"}}]}
                ]
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=prev_message + message
                )

                prev_message = message
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
                    {"role": "user", "content": [
                        {"type": "text", "text": USER_PROMPT}]}
                ]
            )
            structured_transcript = response.choices[0].message.content.replace(
                "```md", "").replace("```", "")
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
                    {"role": "user", "content": [
                        {"type": "text", "text": USER_PROMPT}]}
                ]
            )
            structured_transcript = response.choices[0].message.content.replace(
                "```html", "").replace("```", "")
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
                    {"role": "user", "content": [
                        {"type": "text", "text": USER_PROMPT}]}
                ]
            )
            html_text = response.choices[0].message.content.replace(
                "```html", "").replace("```", "")
        except Exception as e:
            print(f"Error converting markdown to HTML: {e}")
            return None
        return html_text
