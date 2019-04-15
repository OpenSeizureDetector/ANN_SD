import sqlite3
from bottle import route, run, debug, template, request, static_file

parameter_list = ['id', 'name', 'paper_url', 'data_url', 'sensor_pos', 'sample_freq']

############################
#display datasets
############################

@route('/datasets')
def dataset_db():
    conn = sqlite3.connect('dataset_db.db')
    c = conn.cursor()
    c.execute("SELECT {} FROM datasets".format(", ".join(parameter_list)))
    results = c.fetchall()
    c.close()
    output = template('make_table', rows=results)
    return output

############################
#add new dataset
############################

@route('/new', method='GET')
def new_entry():

    if request.GET.save:
        
        conn = sqlite3.connect('dataset_db.db')
        c = conn.cursor()

        new_name = request.GET.name.strip()
        new_paper_url = request.GET.paper_url.strip()
        new_data_url = request.GET.data_url.strip()
        new_sensor_pos = request.GET.sensor_pos.strip()
        new_sample_freq = int(request.GET.sample_freq.strip())
        
        c.execute("INSERT INTO datasets (name, paper_url, data_url, sensor_pos, sample_freq) VALUES (?,?,?,?,?)", (new_name, new_paper_url, new_data_url, new_sensor_pos, new_sample_freq))
        conn.commit()
        c.close()
        return dataset_db()

    else:
        return template('new_entry.tpl')

##############################
#delete dataset
##############################

@route('/delete', method='GET')
def delete_entry():
    
    if request.GET.submit:
        
        conn = sqlite3.connect('dataset_db.db')
        c = conn.cursor()
        entry_to_del = int(request.GET.entry_id)
        c.execute("DELETE FROM datasets WHERE id=?", (entry_to_del,))
        conn.commit()
        c.close()
        return dataset_db()
    
    else:
        return template('delete_entry.tpl')
    
###############################
#include css file
###############################

@route('/static/style.css')
def static(style):
    return static_file(style.css, root='./static')




    
debug(True)
run(reloader=True)
