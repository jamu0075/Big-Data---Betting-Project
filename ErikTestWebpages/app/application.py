from flask import Flask, render_template
import numpy as np
import io
import random
from flask import Response
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure


app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')
    
@app.route('/about/')
def about():
    return render_template('about.html')
   
@app.route('/plot/')
def plot():
# 	print('i got here')
# 	price = [1, 3, 4, 5, 6]
# 	lnprice=np.log(price)
# 	plt.plot(lnprice)

# 	plt.savefig('/static/images/new_plot.png')
	return render_template('plot.html', name = 'new_plot', url ='/static/images/new_plot.png')

@app.route('/plot.png')
def generate_plot():
    fig = create_figure()
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')

def create_figure():
    fig = Figure()
    axis = fig.add_subplot(1, 1, 1)
    xs = range(100)
    ys = [random.randint(1, 50) for x in xs]
    axis.plot(xs, ys)
    return fig


if __name__ == '__main__':
    app.run(port=80)