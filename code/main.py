from flask import Flask, render_template, request, redirect, url_for, g, session
from flask_session import Session

from manage_user import user

import torch

from rnn import *
from util import Util

import pickle
import html
import os
import signal
import subprocess

app = Flask(__name__)
app.secret_key = 'A0Zr98j/3yX R~XHH!jmN]LWX/,?RS'
SESSION_TYPE = 'filesystem'
app.config.from_object(__name__)
Session(app)

description = {
 'title'                            :['professional title', 'KKEYWORDD is/was a AANSWERR.'],
 'students'                         :['student', 'AANSWERR is/was a student of KKEYWORDD.'],
 'schools_attended'                 :['studied at', 'KKEYWORDD is studying/studied at AANSWERR.'],
 'number_of_employees_members'      :['number of emloyees/members', 'KKEYWORDD has AANSWERR employees/members.'],
 'top_employee_or_members'          :['top employee/member', 'AANSWERR is/was in high-level, leading positions as a employee/member of KKEYWORDD.'],
 'top_members_employees'          :['top employee/member', 'AANSWERR is/was in high-level, leading positions as a employee/member of KKEYWORDD.'],
 'top_employee_or_member_of'        :['works for as top', 'KKEYWORDD works/worked for AANSWERR in high-level, leading positions.'],
 'employees_or_members'              :['employee or member', 'AANSWERR is/was a employee/member of KKEYWORDD.'],
 'employee_or_member_of'            :['works for', 'KKEYWORDD works/worked for AANSWERR.'],
 'employee_of'                      :['works for', 'KKEYWORDD works/worked for AANSWERR.'],
 'member_of'                        :['member of', 'KKEYWORDD is/was a member of AANSWERR, though KKEYWORDD can operate independently of AANSWERR.'],
 'members'                          :['has member', 'KKEYWORDD has/had AANSWERR as a member, though AANSWERR can operate independently of KKEYWORDD.'],
 'holds_shares_in'                   :['holds shares in', 'KKEYWORDD holds/held shares in AANSWERR.'],
 'shareholders'                     :['shareholder', 'AANSWERR is/was a shareholder of KKEYWORDD.'],
 'founded_by'                       :['founded by', 'KKEYWORDD was founded by AANSWERR.'],
 'organizations_founded'            :['founded', 'KKEYWORDD founded AANSWERR.'],
 'date_founded'                     :['founded on', 'KKEYWORDD was founded on AANSWERR.'],
 'date_dissolved'                   :['dissolved on', 'KKEYWORDD was closed/dissolved on AANSWER.'],
 'date_of_birth'                    :['born on', 'KKEYWORDD born on AANSWERR.'],
 'date_of_death'                    :['died on', 'KKEYWORDD died on AANSWERR.'],
 'births_in_city'                   :['born at', 'The person who was born at KKEYWORDD is AANSWERR.'],
 'births_in_stateorprovince'        :['born at', 'The person who was born at KKEYWORDD is AANSWERR.'],
 'births_in_country'                :['born at', 'The person who was born at KKEYWORDD is AANSWERR.'],
 'deaths_in_city'                   :['died at', 'The person who died at KKEYWORDD is AANSWERR.'],
 'deaths_in_stateorprovince'        :['died at', 'The person who died at KKEYWORDD is AANSWERR.'],
 'deaths_in_country'                :['died at', 'The person who died at KKEYWORDD is AANSWERR.'],
 'residents_of_city'                :['lived at', 'The person who lives/lived at KKEYWORDD is AANSWERR.'],
 'residents_of_stateorprovince'     :['lived at', 'The person who lives/lived at KKEYWORDD is AANSWERR.'],
 'residents_of_country'             :['lived at', 'The person who lives/lived at KKEYWORDD is AANSWERR.'],
 'headquarters_in_city'             :['headquartered at', 'The entity that is/was headquartered at KKEYWORDD is AANSWERR.'],
 'headquarters_in_stateorprovince'  :['headquartered at', 'The entity that is/was headquartered at KKEYWORDD is AANSWERR.'],
 'headquraters_in_country'          :['headquartered at', 'The entity that is/was headquartered at KKEYWORDD is AANSWERR.'],
 'country_of_birth'                 :['born at', 'KKEYWORDD was born at AANSWERR.'],
 'stateorprovince_of_birth'         :['born at', 'KKEYWORDD was born at AANSWERR.'],
 'city_of_birth'                    :['born at', 'KKEYWORDD was born at AANSWERR.'],
 'country_of_death'                 :['died at', 'KKEYWORDD died at AANSWERR.'],
 'stateorprovince_of_death'         :['died at', 'KKEYWORDD died at AANSWERR.'],
 'city_of_death'                    :['died at', 'KKEYWORDD died at AANSWERR.'],
 'countries_of_residence'           :['lived at', 'KKEYWORDD lives/lived at AANSWERR.'],
 'statesorprovinces_of_residence'   :['lived at', 'KKEYWORDD lives/lived at AANSWERR.'],
 'cities_of_residence'              :['lived at', 'KKEYWORDD lives/lived at AANSWERR.'],
 'children'                         :['child of', 'AANSWERR is the child of KKEYWORDD.'],
 'city_of_headquarters'             :['headquartered at', 'KKEYWORDD is/was headquartered at AANSWERR.'],
 'stateorprovince_of_headquarters'  :['headquartered at', 'KKEYWORDD is/was headquartered at AANSWERR.'],
 'country_of_headquarters'          :['headquartered at', 'KKEYWORDD is/was headquartered at AANSWERR.'],
 'subsidiaries'                     :['subsidary of', 'KKEYWORDD is/was a subsidiary of AANSWERR and KKEYWORDD can not exist without AANSWERR.'],
 'org:parents'                       :['parent of', 'KKEYWORDD owns/owned AANSWERR and AANSWERR can not exist without KKEYWORDD'],
 'per:parents'                       :['parent of', 'AANSWERR is the parent of KKEYWORDD.'],
 'spouse'                           :['spouse', 'AANSWERR is/was the spouse of KKEYWORDD.'],
 'sibling'                          :['sibling of', 'AANSWERR is the sibling of KKEYWORDD.'],
 'other_family'                     :['other family', 'KKEYWORDD and AANSWERR are otherwise family.'],
 'age'                              :['age', 'KKEYWORDD is AANSWERR years old.'],
 'alternate_names'                  :['altername name', 'AANSWERR is an alternate name of KKEYWORDD.'],
 'website'                          :['website', 'A website address of KKEYWORDD is AANSWERR.'],
 'cause_of_death'                    :['cause of death', 'AANSWERR was an explicit cause of death for KKEYWORDD.'],
 'political_religious_affiliation'  :['political/religious affiliation', 'KKEYWORDD is politically or religiously affliated with AANSWERR.'],
 'religion'                         :['religion', 'KKEYWORDD has religiously belonged to AANSWERR.'],
 'origin'                           :['nationality/ethnicity', 'AANSWERR is a nationality or ethnicity of KKEYWORDD.'],
 'charges'                          :['charges', 'KKEYWORDD is/was convicted of AANSWERR.'],
 'no_relation'                          :['no relation', 'There is no relation between KKEYWORDD and AANSWERR,'],
}

MODEL_PATH = "../user/%s/model/"

MAX_BATCH_SIZE = 500

USE_GPU = False

with open("./relation_list.txt") as f:
    tmp = f.read().split("\n")

    if tmp[-1] == "":
        tmp = tmp[:-1]

    relation_list = []

    for i in tmp:
        if i[0] != "#":
            relation_list.append(i)

    RELATION_DICT = {relation:i for i, relation in enumerate(relation_list)}

with open("../data/ANGELIS_POSITION.pkl","rb") as f:
    data = pickle.load(f)

sentence = []
entity = []
entity_position = []
filler = []
filler_position = []
relation = []

for sent, ent, entity_begin, entity_end, fil, filler_begin, filler_end, rel, confidence in data:
    if (rel != "no_relation") & (rel not in RELATION_DICT):
        continue
    sentence.append(html.unescape(sent.lower()))

    splited_sent = html.unescape(sent.lower()).split(" ")

    entity.append(ent.strip())
    filler.append(fil.strip())

    tmp_ent_pos = []
    for i in range(len(splited_sent)):
        if i < entity_begin:
            tmp_ent_pos.append(i - entity_begin)
        elif i < (entity_end - 1):
            tmp_ent_pos.append(0)
        else:
            tmp_ent_pos.append(i - (entity_end - 1))
    entity_position.append(tmp_ent_pos)

    tmp_fil_pos = []
    for i in range(len(splited_sent)):
        if i < filler_begin:
            tmp_fil_pos.append(i - filler_begin)
        elif i < (filler_end - 1):
            tmp_fil_pos.append(0)
        else:
            tmp_fil_pos.append(i - (filler_end - 1))
    filler_position.append(tmp_fil_pos)

    relation.append(rel)

print("# of data:", len(sentence))

VALID_START = int(0.7 * len(sentence))

train_sentence = sentence[:VALID_START]
train_entity = entity[:VALID_START]
train_entity_position = entity_position[:VALID_START]
train_filler = filler[:VALID_START]
train_filler_position = filler_position[:VALID_START]
train_relation = relation[:VALID_START]

valid_sentence = sentence[VALID_START:]
valid_entity = entity[VALID_START:]
valid_entity_position = entity_position[VALID_START:]
valid_filler = filler[VALID_START:]
valid_filler_position = filler_position[VALID_START:]
valid_relation = relation[VALID_START:]

util = Util(train_sentence, "./relation_list.txt", 3, 10)

def get_user():
    if "current_user" not in session:
        current_user = user()
    else:
        current_user = session["current_user"]
    print(current_user.name)
    return current_user

@app.route('/')
@app.route('/index.html')
def index():
    current_user = get_user()
    if current_user.is_active == False:
        return render_template("index.html", user = current_user)
    else:
        instance_data = train_sentence[current_user.cursor]
        if relation[current_user.cursor] == "no_relation":
            instance_description =  description[train_relation[current_user.cursor]][1].replace("KKEYWORDD", "<span style='color:red;font-weight:bold;'>%s</span>" % train_entity[current_user.cursor]).replace("AANSWERR", "<span style='color:blue;font-weight:bold;'>%s</span>" % train_filler[current_user.cursor])
        else:
            instance_description =  description[train_relation[current_user.cursor][4:]][1].replace("KKEYWORDD", "<span style='color:red;font-weight:bold;'>%s</span>" % train_entity[current_user.cursor]).replace("AANSWERR", "<span style='color:blue;font-weight:bold;'>%s</span>" % train_filler[current_user.cursor])
        return render_template("index.html", user = current_user, data = instance_data, description = instance_description, statistics = current_user.statistics(util))

@app.route('/', methods = ["POST"])
@app.route('/index.html', methods = ["POST"])
def index_post():
    current_user = get_user()

    instance_data = train_sentence[current_user.cursor]
    instance_e1 = train_entity[current_user.cursor]
    instance_e2 = train_filler[current_user.cursor]
    instance_e1_position = train_entity_position[current_user.cursor]
    instance_e2_position = train_filler_position[current_user.cursor]
    instance_relation = train_relation[current_user.cursor]
    if relation[current_user.cursor] == "no_relation":
        instance_description =  description[train_relation[current_user.cursor]][1].replace("KKEYWORDD", "<span style='color:red;font-weight:bold;'>%s</span>" % train_entity[current_user.cursor]).replace("AANSWERR", "<span style='color:blue;font-weight:bold;'>%s</span>" % train_filler[current_user.cursor])
    else:
        instance_description =  description[train_relation[current_user.cursor][4:]][1].replace("KKEYWORDD", "<span style='color:red;font-weight:bold;'>%s</span>" % train_entity[current_user.cursor]).replace("AANSWERR", "<span style='color:blue;font-weight:bold;'>%s</span>" % train_filler[current_user.cursor])

    if (request.form['labeling'] == "Yes")|(request.form['labeling'] == "No"):
        if request.form['labeling'] == "Yes":
            label = 1
        else:
            label = 0
        current_user.update(instance_data, instance_e1, instance_e2, instance_e1_position, instance_e2_position, instance_relation, label)

        instance_data = train_sentence[current_user.cursor]
        if relation[current_user.cursor] == "no_relation":
            instance_description =  description[train_relation[current_user.cursor]][1].replace("KKEYWORDD", "<span style='color:red;font-weight:bold;'>%s</span>" % train_entity[current_user.cursor]).replace("AANSWERR", "<span style='color:blue;font-weight:bold;'>%s</span>" % train_filler[current_user.cursor])
        else:
            instance_description =  description[train_relation[current_user.cursor][4:]][1].replace("KKEYWORDD", "<span style='color:red;font-weight:bold;'>%s</span>" % train_entity[current_user.cursor]).replace("AANSWERR", "<span style='color:blue;font-weight:bold;'>%s</span>" % train_filler[current_user.cursor])

        if (current_user.cursor > 10) == 0:
            pass

    return render_template("index.html", user = current_user, data = instance_data, description = instance_description, statistics = current_user.statistics(util), data_vis = None)

@app.route('/login.html')
def login():
    return render_template("login.html")

@app.route('/login.html', methods = ["POST"])
def login_post():
    name =  request.form['name']
    password = request.form['password']
    current_user = user()
    check, message = current_user.login(name, password)
    print(message)

    if check == 1:
        #return index(current_user)
        session['current_user'] = current_user
        
        if os.path.exists(MODEL_PATH % name) is False:
            os.mkdir(MODEL_PATH % name)

            word_emb = Word_Embedding(150, util, USE_GPU)
            entity_position_emb = Position_Embedding(25, util, USE_GPU)
            filler_position_emb = Position_Embedding(25, util, USE_GPU)
            sentence_emb = Sentence_Embedding(word_emb, entity_position_emb, filler_position_emb, util, USE_GPU)

            classifier = Classifier(sentence_emb, 200, util, USE_GPU)
            
            classifier.save(name)
        else:
            if os.path.exists("%s/latest" % (MODEL_PATH % name)) is False:
                word_emb = Word_Embedding(150, util, USE_GPU)
                entity_position_emb = Position_Embedding(25, util, USE_GPU)
                filler_position_emb = Position_Embedding(25, util, USE_GPU)
                sentence_emb = Sentence_Embedding(word_emb, entity_position_emb, filler_position_emb, util, USE_GPU)

                classifier = Classifier(sentence_emb, 200, util, USE_GPU)

                classifier.save(name)
        
        return redirect(url_for("index"))

@app.route('/logout.html')
def logout():
    session.pop('current_user', None)
    return redirect(url_for("index"))

if __name__ == '__main__':

    app.debug = True
    app.run(host='0.0.0.0')
