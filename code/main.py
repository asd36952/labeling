from manage_user import user

from flask import Flask, render_template, request, redirect, url_for, g, session
from flask_session import Session

import pickle
import html

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
 'member_of'                        :['member of', 'KKEYWORDD is/was a member of AANSWERR, though KKWORDD can operate independently of AANSWERR.'],
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
 'age'                              :['age', 'KKEYWORDD is AANSWER years old.'],
 'alternate_names'                  :['altername name', 'AANSWERR is an alternate name of KKEYWORDD.'],
 'website'                          :['website', 'A website address of KKEYWORDD is AANSWERR.'],
 'cause_of_death'                    :['cause of death', 'AANSWERR was an explicit cause of death for KKEYWORDD.'],
 'political_religious_affiliation'  :['political/religious affiliation', 'KKEYWORDD is politically or religiously affliated with AANSWERR.'],
 'religion'                         :['religion', 'KKEYWORDD has religiously belonged to AANSWERR.'],
 'origin'                           :['nationality/ethnicity', 'AANSWERR is a nationality or ethnicity of KKEYWORDD.'],
 'charges'                          :['charges', 'KKEYWORDD is/was convicted of AANSWER.'],
}

with open("../data/ANGELIS_POSITION.pkl","rb") as f:
    data = pickle.load(f)

def get_user():
    if "current_user" not in session:
        current_user = user()
    else:
        current_user = session["current_user"]
    print(current_user.name)
    return current_user

def data_generate(data, cursor):
    sent, e1, e1_start, e1_end, e2, e2_start, e2_end, rel, conf = data[cursor]
    return sent.strip(), description[rel[4:]][1].replace("KKEYWORDD", "<span style='color:red;font-weight:bold;'>%s</span>"%e1.strip()).replace("AANSWERR", "<span style='color:blue;font-weight:bold;'>%s</span>"%e2.strip())

@app.route('/')
@app.route('/index.html')
def index():
    current_user = get_user()
    if current_user.is_active == False:
        return render_template("index.html", user = current_user, data = None, description = None)
    else:
        instance_data, instance_description = data_generate(data, current_user.cursor)
        return render_template("index.html", user = current_user, data = instance_data, description = instance_description)

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
        return redirect(url_for("index"))

@app.route('/logout.html')
def logout():
    session.pop('current_user', None)
    return redirect(url_for("index"))
"""
@app.route('/labeling.html')
def labeling():
    return render_template("labeling.html", user = user())
"""

"""
@app.route('/pages')
def pages_index():
    return render_template("pages/index.html")
"""

"""
@app.route('/<pagename>')
def admin(pagename):
    print(pagename)
    return render_template("pages/" + pagename)
"""

if __name__ == '__main__':

    app.debug = True
    app.run(host='0.0.0.0')
