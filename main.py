from transformers import AutoModelForCausalLM, AutoTokenizer # type: ignore
import torch
import requests
import selenium
import re
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
from pygments.lexers import PythonLexer
from pygments import lex
from transformers import (
    AutoModelForSeq2SeqLM, # type: ignore
    AutoTokenizer, # type: ignore
    AutoConfig, # type: ignore
    pipeline, # type: ignore
)
import os
import azure.cognitiveservices.speech as speechsdk
from transformers import BartTokenizer, BartForConditionalGeneration # type: ignore

class WebScraper:
    def __init__(self) -> None:
        self.subscription_key = ""

    def search_articles(self, query):
        subscription_key = ''
        endpoint = 'https://api.bing.microsoft.com' + "/v7.0/search"

        headers = {"Ocp-Apim-Subscription-Key" : subscription_key}
        params  = {"q": query, "textDecorations": True, "textFormat": "HTML"}
        response = requests.get(endpoint, headers=headers, params=params)
        response.raise_for_status()
        search_results = response.json()

        # Get the URLs of the articles
        urls = [result['url'] for result in search_results['webPages']['value']]

        return urls

    def Get_webpage_text(self, urls):
        texts = []
        # Setup ChromeDriver
        serv = Service(executable_path="D:\\CODE\\SRC\\chromedriver_win32\\chromedriver.exe")
        # Setup ChromeDriver
        driver = webdriver.Chrome(service=serv)
        for url in urls:
            # Navigate to the URL
            driver.get(url)

            # Get the page source
            page_source = driver.page_source

            # Get text from the body of the webpage
            element = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            text = driver.find_element(By.TAG_NAME, "body").text

            # Break into lines and remove leading and trailing space on each
            lines = (line.strip() for line in text.splitlines())

            # Break multi-headlines into a line each
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))

            # Drop blank lines
            text = '\n'.join(chunk for chunk in chunks if chunk)

            texts.append(text)

        # Close the browser
        driver.quit()

        return texts


    def preprocess_text(self, text):
        # Remove special characters
        text = re.sub(r'\W', ' ', text)

        # Remove digits
        text = re.sub(r'\d', ' ', text)

        # Remove single characters
        text = re.sub(r'\s+[a-zA-Z]\s+', ' ', text)

        # Remove multiple spaces
        text = re.sub(r'\s+', ' ', text, flags=re.I)

        # Convert to lowercase
        text = text.lower()

        return text

class RecognitionService:
    def __init__(self) -> None:
        self.speech_config = speechsdk.SpeechConfig(subscription="", region="")
    def Speack(self, content):
        # This example requires environment variables named "SPEECH_KEY" and "SPEECH_REGION"
        speech_config = speechsdk.SpeechConfig(subscription="", region="")
        audio_config = speechsdk.audio.AudioOutputConfig(use_default_speaker=True) # type: ignore

        # The language of the voice that speaks.
        speech_config.speech_synthesis_voice_name='en-US-SteffanNeural'

        speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)

        # Get text from the console and synthesize to the default speaker.
        print("Enter some text that you want to speak >")
        text = content

        speech_synthesis_result = speech_synthesizer.speak_text_async(text).get()

        if speech_synthesis_result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted: # type: ignore
            print("Speech synthesized for text [{}]".format(text))
        elif speech_synthesis_result.reason == speechsdk.ResultReason.Canceled: # type: ignore
            cancellation_details = speech_synthesis_result.cancellation_details # type: ignore
            print("Speech synthesis canceled: {}".format(cancellation_details.reason))
            if cancellation_details.reason == speechsdk.CancellationReason.Error:
                if cancellation_details.error_details:
                    print("Error details: {}".format(cancellation_details.error_details))
                    print("Did you set the speech resource key and region values?")
    
    def recognize_from_microphone(self):
        # This example requires environment variables named "SPEECH_KEY" and "SPEECH_REGION"
        speech_config = speechsdk.SpeechConfig(subscription="72dba82887424328a7aa7691d1a1388d", region="eastus")
        speech_config.speech_recognition_language="en-US"

        audio_config = speechsdk.audio.AudioConfig(use_default_microphone=True)
        speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)

        print("Speak into your microphone.")
        self.Speack("Speak into your microphone.")
        speech_recognition_result = speech_recognizer.recognize_once_async().get()

        if speech_recognition_result.reason == speechsdk.ResultReason.RecognizedSpeech: # type: ignore
            print("Recognized: {}".format(speech_recognition_result.text)) # type: ignore
            return speech_recognition_result.text # type: ignore
        elif speech_recognition_result.reason == speechsdk.ResultReason.NoMatch: # type: ignore
            print("No speech could be recognized: {}".format(speech_recognition_result.no_match_details)) # type: ignore
        elif speech_recognition_result.reason == speechsdk.ResultReason.Canceled: # type: ignore
            cancellation_details = speech_recognition_result.cancellation_details # type: ignore
            print("Speech Recognition canceled: {}".format(cancellation_details.reason))
            if cancellation_details.reason == speechsdk.CancellationReason.Error:
                print("Error details: {}".format(cancellation_details.error_details))
                print("Did you set the speech resource key and region values?")


class SearchInfo:
    def __init__(self) -> None:
        self.output = RecognitionService().recognize_from_microphone()
        self.url = WebScraper().search_articles(str(self.output))
        self.data = WebScraper().Get_webpage_text(str(self.url))
    def model(self):
        data2 = WebScraper().preprocess_text(str(self.data))


        # Load pre-trained model and tokenizer
        model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
        tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')

        # Define the text we want to summarize
        text = data2

        # Tokenize the input text
        input_tokens = tokenizer.encode(text, return_tensors='pt', max_length=1024, truncation=True)

        # Generate a summary of the text
        summary_ids = model.generate(input_tokens, num_beams=4, max_length=100, early_stopping=True)

        # Decode the summary
        summary_text = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

        print(summary_text)

class CodeSearch:
    def __init__(self) -> None:
        pass
    def search_articles(self, query):
        subscription_key = 'aab5d93e331f44a3b7a15b71356cc064'
        endpoint = 'https://api.bing.microsoft.com' + "/v7.0/search"

        headers = {"Ocp-Apim-Subscription-Key" : subscription_key}
        params  = {"q": query, "textDecorations": True, "textFormat": "HTML"}
        response = requests.get(endpoint, headers=headers, params=params)
        response.raise_for_status()
        search_results = response.json()

        # Get the URLs of the articles
        urls = [result['url'] for result in search_results['webPages']['value']]

        return urls

    def Get_webpage_text(self, urls):
        texts = []
        serv = Service(executable_path="D:\\CODE\\SRC\\chromedriver_win32\\chromedriver.exe")
        # Setup ChromeDriver
        driver = webdriver.Chrome(service=serv)
        for url in urls:
            try:
                print(f"Searching for {url}")
                driver.get(url)

                # Get the page source
                page_source = driver.page_source

                # Get text from the body of the webpage
                element = WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.TAG_NAME, "body"))
                )
                text = driver.find_element(By.TAG_NAME, "body").text

                # Break into lines and remove leading and trailing space on each
                lines = (line.strip() for line in text.splitlines())

                # Break multi-headlines into a line each
                chunks = (phrase.strip() for line in lines for phrase in line.split("  "))

                # Drop blank lines
                text = '\n'.join(chunk for chunk in chunks if chunk)

                texts.append(text)
                print("Done")
            except Exception as e:
                print(f"A technical error ocured: {e}")
                pass

        # Close the browser
        print("finishing...")
        driver.quit()

        return texts

    def preprocess_text(self, text):
        # Convert to lowercase
        text = text.lower()

        # Split the text into lines
        lines = text.split('\n')

        # Initialize a Python lexer
        lexer = PythonLexer()

        # Keep only lines that are valid Python code
        python_lines = [line for line in lines if list(lex(line, lexer))]

        # Join the Python lines back into a single string
        python_code = '\n'.join(python_lines)

        return python_code
    def preprocess_text_2(self, text):
        # Remove special characters
        text = re.sub(r'\W', ' ', text)

        # Remove digits
        text = re.sub(r'\d', ' ', text)

        # Remove single characters
        text = re.sub(r'\s+[a-zA-Z]\s+', ' ', text)

        # Remove multiple spaces
        text = re.sub(r'\s+', ' ', text, flags=re.I)

        # Convert to lowercase
        text = text.lower()

        return text

class CodeHelper:
    def __init__(self) -> None:
        pass
    def ExplainGitHubCode(self, data):
        model_name = "sagard21/python-code-explainer"

        tokenizer = AutoTokenizer.from_pretrained(model_name, padding=True)

        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

        config = AutoConfig.from_pretrained(model_name)

        model.eval()

        pipe = pipeline("summarization", model=model_name, config=config, tokenizer=tokenizer)

        urls = CodeSearch().search_articles(data)
        data = CodeSearch().preprocess_text_2(str(CodeSearch().Get_webpage_text(urls)))

        raw_code = CodeSearch().preprocess_text(data)
        raw_code = raw_code[:10000]

        return pipe(raw_code)[0]["summary_text"]
    def CodeGen(self, data):
        prompt = "class GetText:"
        q1 = data

        PromptToData = f"""
        '''{q1}''' \n
        {prompt}
        """

        tokenizer = AutoTokenizer.from_pretrained('replit/replit-code-v1-3b', trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained('replit/replit-code-v1-3b', trust_remote_code=True)

        x = tokenizer.encode(PromptToData, return_tensors='pt')
        y = model.generate(x, max_length=500, do_sample=True, top_p=0.95, top_k=4, temperature=0.7, num_return_sequences=1, eos_token_id=tokenizer.eos_token_id)

        # decoding, clean_up_tokenization_spaces=False to ensure syntactical correctness
        generated_code = tokenizer.decode(y[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
        print(generated_code)

def main():
    data3 = RecognitionService().recognize_from_microphone()
    if str(data3).startswith("Make"):
        print("Generating Code")
        CodeHelper().CodeGen(data3)
    if str(data3).startswith("Find"):
        print("Finding code to explain")
        CodeHelper().ExplainGitHubCode(data3)
    if str(data3).startswith("Search") or str(data3).startswith("Information") or str(data3).startswith("For"):
        print("Searching for info")
        # data = preprocess_text_2(str(Get_webpage_text(urls)))
        SearchInfo().model()


if __name__ == '__main__':
    main()
