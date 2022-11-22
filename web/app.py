from flask import Flask, jsonify, request
from flask_restful import Api, Resource
from pymongo import MongoClient
import bcrypt
import language_tool_python
# using the tool  
my_tool = language_tool_python.LanguageTool('en-US')  
import spacy
nlp = spacy.load('en_core_web_sm')

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

import spacy
nlp = spacy.load('en_core_web_sm')

client = MongoClient("mongodb://localhost:27017")
db = client.SimilarityDB
users = db["Users"]

base_path=os.getcwd()

def UserExist(username):
    # if users.find({"Username":username}).count() == 0:
    if users.count_documents({"Username":username}) == 0:
        return False
    else:
        return True

class Register(Resource):
    def post(self):
        #Step 1 is to get posted data by the user
        postedData = request.get_json()

        #Get the data
        username = postedData["username"]
        password = postedData["password"] #"123xyz"

        if UserExist(username):
            retJson = {
                'status':301,
                'msg': 'Invalid Username'
            }
            return jsonify(retJson)

        hashed_pw = bcrypt.hashpw(password.encode('utf8'), bcrypt.gensalt())

        #Store username and pw into the database
        users.insert_one({
            "Username": username,
            "Password": hashed_pw,
            "Tokens":6
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
        "Username":username
    })[0]["Password"]

    if bcrypt.hashpw(password.encode('utf8'), hashed_pw) == hashed_pw:
        return True
    else:
        return False

def countTokens(username):
    tokens = users.find({
        "Username":username
    })[0]["Tokens"]
    return tokens

class Detect(Resource):
    def post(self):
        #Step 1 get the posted data
        postedData = request.get_json()

        #Step 2 is to read the data
        username = postedData["username"]
        password = postedData["password"]
        path1 = postedData["path1"]
        path1=base_path+"\\documents"+"\\"+path1
        path2 = postedData["path2"]
        path2=base_path+"\\documents"+"\\"+path2

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

        #Calculate edit distance between text1, text2
        import spacy
        nlp = spacy.load('en_core_web_sm')
        text1 = nlp(text1)
        text2 = nlp(text2)

        ratio = text1.similarity(text2)

        retJson = {
            "status":200,
            "ratio": ratio,
            "msg":"Similarity score calculated successfully"
        }

        #Take away 1 token from user
        current_tokens = countTokens(username)
        users.update_one({
            "Username":username
        }, {
            "$set":{
                "Tokens":int(current_tokens)-1
                }
        })

        return jsonify(retJson)

class Grammer_Check(Resource):
    def post(self):
        #Step 1 get the posted data
        postedData = request.get_json()
        #Step 2 is to read the data
        username = postedData["username"]
        password = postedData["password"]
        text = postedData["text"]
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
        
        # correction  
        correct_text = my_tool.correct(text)  

        retJson = {
            "status":200,
            "msg":"Text Corrected Successfully !!",
            "Original Text": text,
            "Text after correction": correct_text
        }

        #Take away 1 token from user
        current_tokens = countTokens(username)
        users.update({
            "Username":username
        }, {
            "$set":{
                "Tokens":int(current_tokens)-1
                }
        })

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
                "status":304,
                "msg": "Invalid Admin Password"
            }
            return jsonify(retJson)

        #MAKE THE USER PAY!
        users.update_one({
            "Username":username
        }, {
            "$set":{
                "Tokens":refill_amount
                }
        })

        retJson = {
            "status":200,
            "msg": "Refilled successfully"
        }
        return jsonify(retJson)

class Summarize(Resource):
    def post(self):
        posted_data=request.get_json()
        username=posted_data["username"]
        password=posted_data["password"]
        document_path=posted_data["path"]
        number_of_sentences=int(posted_data["sentences"])
        # verify username
        if not UserExist(username):
            retJson={"status":301,
            "msg":"Invalid Username"
            }
            return jsonify(retJson)
        # verify password
        correct_pw=verifyPw(username,password)
        
        if not correct_pw:
            retJson={"status":302,
                     "msg":"Incorrect Password"
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
        text,summarized_text=summarize_text(document_path,number_of_sentences)
        if text=="" and summarized_text=="":
            retJson={"status":304,
                     "msg":"File not found"
                     }
            return jsonify(retJson)
        elif len(summarized_text)==0 and len(text)>0:
            retJson={"status":305,
                     "msg":"Please provide sufficient number to summarize text"
                     }
            return jsonify(retJson)
        retJson={"text":text,
                 "summarized_text":summarized_text
                 }
        #Take away 1 token from user
        current_tokens = countTokens(username)
        users.update_one({
            "Username":username
        }, {
            "$set":{
                "Tokens":int(current_tokens)-1
                }
        })
        return jsonify(retJson)

class ReadingTime(Resource):
    def post(self):
        posted_data=request.get_json()
        username=posted_data["username"]
        password=posted_data["password"]
        document_path=posted_data["path"]
        document_path=base_path+"\\documents"+"\\"+document_path
        try:
            wpm=posted_data["wpm"]
        except:
            wpm=120
        
        # verify username
        if not UserExist(username):
            retJson={"status":301,
            "msg":"Invalid Username"
            }
            return jsonify(retJson)
        # verify password
        correct_pw=verifyPw(username,password)
        
        if not correct_pw:
            retJson={"status":302,
                     "msg":"Incorrect Password"
                     }
            return jsonify(retJson)
        # verify user has enough tokens
        num_tokens=countTokens(username)
        if int(num_tokens<=0):
            retJson={"status":303,
                     "msg":"You are out of tokens"
                     }
            return jsonify(retJson)
        # calculate reading time finally
        try:
            with open(document_path) as file:
                text=file.read()
        except:
            retJson={"status":304,
                     "msg":"File not found"}
            return jsonify(retJson)
        reading_time=calculate_reading_time(text,wpm)
        retJson={"reading time (minutes)":reading_time}
        return jsonify(retJson)
    
api.add_resource(Register, '/register')
api.add_resource(Similarity, '/similarity')
api.add_resource(Grammer_Check,'/grammer_check')
api.add_resource(Refill, '/refill')
api.add_resource(Summarize,'/summarize')
api.add_resource(ReadingTime,'/readingtime')

if __name__=="__main__":
    app.run(host='0.0.0.0')
