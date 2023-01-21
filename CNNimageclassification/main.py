from flask import Flask

app = Flask(__name__)

print(__name__ + "hello world")


@app.route('/<name>')
def initialize(name):
    return 'Hello world slash %s' %name


def initialize_hai():
    return 'Hello world hai'
app.add_url_rule('/','hello', 'helloworld')

app.run()
