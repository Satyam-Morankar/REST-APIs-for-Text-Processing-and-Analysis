from nltk.tokenize import sent_tokenize, word_tokenize
import os
import re
import nltk
import heapq
from nltk.corpus import stopwords
import spacy
from flask import Flask, jsonify, request
from flask_restful import Api, Resource
from pymongo import MongoClient
import bcrypt
#-------------------------------------------------------
# Grammar check dependencies
import language_tool_python
import requests
# using the tool
my_tool = language_tool_python.LanguageTool('en-US')
# -------------------------------------------------------------
# paraphrase dependencies
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
tokenizer = AutoTokenizer.from_pretrained("Vamsi/T5_Paraphrase_Paws")
model = AutoModelForSeq2SeqLM.from_pretrained("Vamsi/T5_Paraphrase_Paws")
# --------------------------------------------------------------
# Similarity check dependencies
nlp = spacy.load('en_core_web_sm')
import pyttsx3

audio_number=1

bot=pyttsx3.init()
voices=bot.getProperty('voices')
bot.setProperty('rate',125)
bot.setProperty('voice',voices[1].id)
bot.setProperty('volume',1)

# ------------------------------------------
# summarization depecdancies
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize,word_tokenize
import heapq
import nltk
import re
import os
# ------------------------------------------

app = Flask(__name__)
api = Api(app)

client = MongoClient("mongodb://localhost:27017")
db = client.SimilarityDB
users = db["Users"]

base_path = os.getcwd()


def UserExist(username):
    # if users.find({"Username":username}).count() == 0:
    if users.count_documents({"Username": username}) == 0:
        return False
    else:
        return True


class Register(Resource):
    def post(self):
        # Step 1 is to get posted data by the user
        postedData = request.get_json()

        # Get the data
        username = postedData["username"]
        password = postedData["password"]  # "123xyz"

        if UserExist(username):
            retJson = {
                'status': 301,
                'msg': 'Invalid Username'
            }
            return jsonify(retJson)

        hashed_pw = bcrypt.hashpw(password.encode('utf8'), bcrypt.gensalt())

        # Store username and pw into the database
        users.insert_one({
            "Username": username,
            "Password": hashed_pw,
            "Tokens": 6
        })

        retJson = {
            "status": 200,
            "msg": "You have successfully signed up for the API"
        }
        return jsonify(retJson)


def verifyPw(username, password):
    if not UserExist(username):
        return False

    hashed_pw = users.find({
        "Username": username
    })[0]["Password"]

    if bcrypt.hashpw(password.encode('utf8'), hashed_pw) == hashed_pw:
        return True
    else:
        return False


def countTokens(username):
    tokens = users.find({
        "Username": username
    })[0]["Tokens"]
    return tokens


def summarize(text):
    pass


def summarize_text(filename, no_of_sentences=5, print_summarized_text=False):
    # read the text file
    path = base_path+"\\documents"+"\\"+filename
    print(path)
    if not os.path.exists(path):
        retJson = {"status": 304,
                   "msg": "File not found"}
        return jsonify(retJson)
    with open(path) as txt:
        data = txt.read()
    # sentence tokenization
    sentences = sent_tokenize(data)
    print('Total number of sentences in the text: {0}'.format(len(sentences)))

    # cleaning text
    dict = {}
    text = ""
    for a in sentences:
        temp = re.sub("[^a-zA-Z]", " ", a)
        temp = temp.lower()
        dict[temp] = a
        text += temp

    # stop words removal
    stopwords = nltk.corpus.stopwords.words('english')
    word_fre = {}
    for word in nltk.word_tokenize(text):
        if word not in stopwords:
            if word not in word_fre.keys():
                word_fre[word] = 1
            else:
                word_fre[word] += 1

    max_freq = max(word_fre.values())

    # calculating word frequencies in the text
    for w in word_fre:
        word_fre[w] /= max_freq

    # calculating sentence scores based on word frequencies
    sent_scores = {}
    for sent in sentences:
        for word in nltk.word_tokenize(sent.lower()):
            if word in word_fre.keys():
                if len(sent.split(' ')) < 30:
                    if sent not in sent_scores.keys():
                        sent_scores[sent] = word_fre[word]
                    else:
                        sent_scores[sent] += word_fre[word]

    # summary generation using top 'n' sentences from sent_scores
    summarized_sentences = heapq.nlargest(
        no_of_sentences, sent_scores, key=sent_scores.get)
    summarized_text = ' '.join(summarized_sentences)
    if print_summarized_text:
        print('-'*80)
        print('Original text is:')
        print('-'*80)
        print(data)
        print('-'*80)
        print('Summarized text is:')
        print('-'*80)
        print(summarized_text)
        print('-'*80)
    return data, summarized_text


def calculate_reading_time(text, wpm=120):
    tokens = nltk.word_tokenize(text)
    return len(tokens)/wpm


class Similarity(Resource):
    def post(self):
        # Step 1 get the posted data
        postedData = request.get_json()

        # Step 2 is to read the data
        username = postedData["username"]
        password = postedData["password"]
        path1 = postedData["path1"]
        path1 = base_path+"\\documents"+"\\"+path1
        path2 = postedData["path2"]
        path2 = base_path+"\\documents"+"\\"+path2

        if not UserExist(username):
            retJson = {
                'status': 301,
                'msg': "Invalid Username"
            }
            return jsonify(retJson)
        # Step 3 verify the username pw match
        correct_pw = verifyPw(username, password)

        if not correct_pw:
            retJson = {
                "status": 302,
                "msg": "Incorrect Password"
            }
            return jsonify(retJson)
        # Step 4 Verify user has enough tokens
        num_tokens = countTokens(username)
        if int(num_tokens) <= 0:
            retJson = {
                "status": 303,
                "msg": "You are out of tokens, please refill!"
            }
            return jsonify(retJson)

        # Calculate edit distance between text1, text2
        try:
            with open(path1) as file:
                text1 = file.read()
            with open(path2) as file:
                text2 = file.read()
            pass
        except:
            retJson = {"status": 304,
                       "msg": "File not found"
                       }
            return retJson
        # Calculate edit distance between text1, text2
        text1 = nlp(text1)
        text2 = nlp(text2)

        ratio = text1.similarity(text2)

        retJson = {
            "status": 200,
            "ratio": ratio,
            "msg": "Similarity score calculated successfully"
        }

        # Take away 1 token from user
        current_tokens = countTokens(username)
        users.update_one({
            "Username": username
        }, {
            "$set": {
                "Tokens": int(current_tokens)-1
            }
        })

        return jsonify(retJson)


class Grammer_Check(Resource):
    def post(self):
        # Step 1 get the posted data
        postedData = request.get_json()
        # Step 2 is to read the data
        username = postedData["username"]
        password = postedData["password"]
        text = postedData["text"]
        if not UserExist(username):
            retJson = {
                'status': 301,
                'msg': "Invalid Username"
            }
            return jsonify(retJson)
        # Step 3 verify the username pw match
        correct_pw = verifyPw(username, password)

        if not correct_pw:
            retJson = {
                "status": 302,
                "msg": "Incorrect Password"
            }
            return jsonify(retJson)
        # Step 4 Verify user has enough tokens
        num_tokens = countTokens(username)
        if int(num_tokens) <= 0:
            retJson = {
                "status": 303,
                "msg": "You are out of tokens, please refill!"
            }
            return jsonify(retJson)

        # correction
        correct_text = my_tool.correct(text)

        retJson = {
            "status": 200,
            "msg": "Text Corrected Successfully !!",
            "Original Text": text,
            "Text after correction": correct_text
        }

        # Take away 1 token from user
        current_tokens = countTokens(username)
        users.update_one({
            "Username": username
        }, {
            "$set": {
                "Tokens": int(current_tokens)-1
            }
        })

        return retJson 

class Paraphrase(Resource):
    def post(self):
        postedData = request.get_json()

        username = postedData["username"]
        password = postedData["password"]
        document_path= postedData["path"]
        document_path= base_path+"\\documents"+"\\"+document_path

        try:
            with open(document_path) as file:
                text=file.read()
            
            #pass
        except:
            retJson={"status":404,
                     "msg":"File not found"
                     }
            return retJson

        if not UserExist(username):
            retJson = {
                'status':301,
                'msg': "Invalid Username"
            }
            return jsonify(retJson)
        #Step 3 verify the username pw match
        correct_pw = verifyPw(username, password)

        if not correct_pw:
            retJson = {
                "status":302,
                "msg": "Incorrect Password"
            }
            return jsonify(retJson)
        #Step 4 Verify user has enough tokens
        num_tokens = countTokens(username)
        if int(num_tokens) <= 0:
            retJson = {
                "status": 303,
                "msg": "You are out of tokens, please refill!"
            }
            return jsonify(retJson)

        # Step 5: Paraphrase the text:
        def my_paraphrase(sentence):
            sentence = "paraphrase: "+ sentence+" </s>"
            encoding = tokenizer.encode_plus(sentence,padding=True, return_tensors="pt")
            input_ids,attention_masks = encoding["input_ids"],encoding["attention_mask"]

            outputs = model.generate(
                input_ids = input_ids, attention_mask=attention_masks,
                max_length = 256,
                do_sample = True,
                top_k=120,
                top_p=0.95,
                early_stopping=True,
                num_return_sequences=1
            )
            output = tokenizer.decode(outputs[0],skip_special_tokens=True,clean_up_tokenization_spaces=True)

            return(output)

        # join the paraphrased sentences:

        output = " ".join([my_paraphrase(sent) for sent in sent_tokenize(text)])

        # Return json response:

        retJson = {
            "status":200,
            "msg":"The input text is Pharaphrased Successfully !!",
            "Original Text": text,
            "Pharaphrased Text": output
        }    
        return retJson   
        # printing matches  
        



class Refill(Resource):
    def post(self):
        postedData = request.get_json()

        username = postedData["username"]
        password = postedData["admin_pw"]
        refill_amount = postedData["refill"]

        if not UserExist(username):
            retJson = {
                "status": 301,
                "msg": "Invalid Username"
            }
            return jsonify(retJson)

        correct_pw = "abc123"
        if not password == correct_pw:
            retJson = {
                "status": 304,
                "msg": "Invalid Admin Password"
            }
            return jsonify(retJson)

        # MAKE THE USER PAY!
        users.update_one({
            "Username": username
        }, {
            "$set": {
                "Tokens": refill_amount
            }
        })

        retJson = {
            "status": 200,
            "msg": "Refilled successfully"
        }
        return jsonify(retJson)


class Summarize(Resource):
    def post(self):
        posted_data = request.get_json()
        username = posted_data["username"]
        password = posted_data["password"]
        document_path = posted_data["path"]
        try:
            number_of_sentences = int(posted_data["sentences"])
        except:
            number_of_sentences=5
        try:
            speak_text=posted_data["speak"]
        except:
            speak_text="False"
        try:
            save_audio=posted_data["save_audio"]
        except:
            save_audio="False"
        # verify username
        if not UserExist(username):
            retJson = {"status": 301,
                       "msg": "Invalid Username"
                       }
            return jsonify(retJson)
        # verify password
        correct_pw = verifyPw(username, password)

        if not correct_pw:
            retJson = {"status": 302,
                       "msg": "Incorrect Password"
                       }
            return jsonify(retJson)
        # verify user has enough tokens
        num_tokens=countTokens(username)
        if int(num_tokens<=0):
            retJson={"status":303,
                     "msg":"You are out of tokens"
                     }
            return jsonify(retJson)

        # summarize the text finally
        text, summarized_text = summarize_text(
            document_path, number_of_sentences)
        if text == "" and summarized_text == "":
            retJson = {"status": 304,
                       "msg": "File not found"
                       }
            return jsonify(retJson)
        elif len(summarized_text) == 0 and len(text) > 0:
            retJson = {"status": 305,
                       "msg": "Please provide sufficient number to summarize text"
                       }
            return jsonify(retJson)
        retJson={"text":text,
                 "summarized_text":summarized_text
                 }
        #Take away 1 token from user
        current_tokens = countTokens(username)
        users.update_one({
            "Username": username
        }, {
            "$set": {
                "Tokens": int(current_tokens)-1
            }
        })
        if save_audio=="True":
            bot.save_to_file(summarized_text,base_path+"\\audio\\"+str(audio_number)+".mp3")
            audio_number+=1
            
        if speak_text=="True":
            bot.say(summarized_text)
            bot.runAndWait()
        return jsonify(retJson)


class ReadingTime(Resource):
    def post(self):
        posted_data = request.get_json()
        username = posted_data["username"]
        password = posted_data["password"]
        document_path = posted_data["path"]
        document_path = base_path+"\\documents"+"\\"+document_path
        try:
            wpm = posted_data["wpm"]
        except:
            wpm = 120

        # verify username
        if not UserExist(username):
            retJson = {"status": 301,
                       "msg": "Invalid Username"
                       }
            return jsonify(retJson)
        # verify password
        correct_pw = verifyPw(username, password)

        if not correct_pw:
            retJson = {"status": 302,
                       "msg": "Incorrect Password"
                       }
            return jsonify(retJson)
        # verify user has enough tokens
        num_tokens = countTokens(username)
        if int(num_tokens <= 0):
            retJson = {"status": 303,
                       "msg": "You are out of tokens"
                       }
            return jsonify(retJson)
        # calculate reading time finally
        try:
            with open(document_path) as file:
                text = file.read()
        except:
            retJson = {"status": 304,
                       "msg": "File not found"}
            return jsonify(retJson)
        reading_time = calculate_reading_time(text, wpm)
        retJson = {"reading time (minutes)": reading_time}
        return jsonify(retJson)

class Summarize_Similarity(Resource):
    def post(self):
        posted_data=request.get_json()
        username=posted_data["username"]
        password=posted_data["password"]
        file1name=posted_data["path1"]
        file2name=posted_data["path2"]
        try:
            no_of_sentences=int(posted_data["sentences"])
        except:
            no_of_sentences=5
        
        # verify username
        if not UserExist(username):
            retJson = {"status": 301,
                       "msg": "Invalid Username"
                       }
            return jsonify(retJson)
        # verify password
        correct_pw = verifyPw(username, password)

        if not correct_pw:
            retJson = {"status": 302,
                       "msg": "Incorrect Password"
                       }
            return jsonify(retJson)
        # verify user has enough tokens
        num_tokens = countTokens(username)
        if int(num_tokens <= 0):
            retJson = {"status": 303,
                       "msg": "You are out of tokens"
                       }
            return jsonify(retJson)
        # summarize both the texts
        text1,summarized_text1=summarize_text(file1name,no_of_sentences)
        text2,summarized_text2=summarize_text(file2name,no_of_sentences)
        with open(base_path+"\\documents"+"\\summary1.txt","w") as file:
            file.write(summarized_text1)
        with open(base_path+"\\documents"+"\\summary2.txt","w") as file:
            file.write(summarized_text2)
        path1="summary1.txt"
        path2="summary2.txt"
        json_body={"username":username,
                   "password":password,
                   "path1":path1,
                   "path2":path2
                   }
        response = requests.post('http://localhost:5000/similarity',json=json_body)
        if response.status_code !=200:
            retJson={"status":response.status_code,
                     "msg":"Error"
                     }
            return jsonify(retJson)
        return (response.json())
    
    
api.add_resource(Register, '/register')
api.add_resource(Similarity, '/similarity')
api.add_resource(Grammer_Check, '/grammer_check')
api.add_resource(Refill, '/refill')
api.add_resource(Summarize, '/summarize')
api.add_resource(ReadingTime, '/readingtime')
api.add_resource(Summarize_Similarity,'/summarize_similarity')
api.add_resource(Paraphrase, '/paraphrase')

if __name__ == "__main__":
    bot.say('Server booted')
    bot.runAndWait()
    app.run(host='0.0.0.0')
