import io
from flask import Flask, redirect, url_for, render_template, Response, request, session, send_file
import os
import numpy as np
import matplotlib.pylab as plt
import matplotlib.patches as pat
from matplotlib.collections import PatchCollection
import matplotlib.colors as mcolors
import random
import base64
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib
import skimage
from skimage.transform import PiecewiseAffineTransform, warp
from skimage import data
from numpy import exp,pi,sqrt,cos,sin
from numpy.fft import *
import imageio
from urllib.request import urlopen

matplotlib.use('Agg')

app = Flask(__name__)

app = Flask(__name__)
app.secret_key = "super secret key"

@app.route("/")
def home():
    return render_template("qcfinal.html")
    
@app.route("/flow", methods=["POST","GET"])
def flow():
    if request.method=="POST":
        if 'nbands' in request.form:
            number = request.form.get('nbands', type=int)
            session["number"]=number
            spacing = request.form.get('spacing', type=int)
            session["spacing"]=spacing
            color = request.form.get('color', type=int)
            session["color"]=color
            return redirect(url_for('flow'))
        if 'ka' in request.form:
            ka = request.form.get('ka', type=float)
            session["ka"]=ka
            dp = request.form.get('dp', type=float)
            session["dp"]=dp
            return redirect(url_for('flow', results=generate_plots(ka,dp)))
    else:
        return render_template("flow.html")
    

@app.route("/billiards")
def billiards():
    return render_template("billiards.html")
    
@app.route('/plt.png')
def plt_png():
    if "number" in session:
        number=session["number"]
    else:
        number=9
    
    if "spacing" in session:
        spacing=session["spacing"]
    else:
        spacing=3
        
    if "color" in session:
        color=session["color"]
    else:
        color=1
        
    fig = make_image(number,spacing,color)
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    fig.savefig('init_image.png', dpi=160, facecolor='w',bbox_inches="tight", pad_inches=0)
    return Response(output.getvalue(), mimetype='image/png')
    
def make_image(nb,sp,co):
    cs0=['red','orangered','orange','yellow','lawngreen','springgreen','cyan','dodgerblue','blue','blueviolet','magenta','red','orangered','orange','yellow','lawngreen','springgreen','cyan','dodgerblue','blue','blueviolet','magenta','red','orangered','orange','yellow','lawngreen','springgreen','cyan','dodgerblue','blue','blueviolet','magenta']
    cs1=['springgreen','turquoise','teal','lawngreen','aqua','darkturquoise','deepskyblue','cornflowerblue','royalblue','blue','springgreen','turquoise','teal','lawngreen','aqua','darkturquoise','deepskyblue','cornflowerblue','royalblue','blue','springgreen','turquoise','teal','lawngreen','aqua','darkturquoise','deepskyblue','cornflowerblue','royalblue','blue']
    cs2=['black','dimgray','gray','slategray','silver','cadetblue','skyblue','lightsteelblue','powderblue','black','dimgray','gray','slategray','silver','cadetblue','skyblue','lightsteelblue','powderblue','black','dimgray','gray','slategray','silver','skyblue','lightsteelblue','lightsteelblue','powderblue']
    cs3=['indianred','firebrick','maroon','orangered','chocolate','darkorange','orange','goldenrod','gold','indianred','firebrick','maroon','orangered','chocolate','darkorange','orange','goldenrod','gold','indianred','firebrick','maroon','orangered','chocolate','darkorange','orange','goldenrod','gold']
    cs4=['darkblue','blue','mediumslateblue','blueviolet','darkviolet','mediumorchid','purple','magenta','mediumvioletred','darkblue','blue','mediumslateblue','blueviolet','darkviolet','mediumorchid','purple','magenta','mediumvioletred','darkblue','blue','mediumslateblue','blueviolet','darkviolet','mediumorchid','purple','magenta','mediumvioletred']
         
    colorschemes=[cs0,cs1,cs2,cs3,cs4]
    
    nbands=nb#int(input("number of bands = "))
    spacing=sp#int(input("enter spacing--rows of dots between = "))
    color_scheme_i=co#int(input("color scheme number"))

    scheme_choice=colorschemes[color_scheme_i]
    n=40
    x,y=np.meshgrid(np.arange(n)+1,np.arange(n)+1)
    f=plt.figure(figsize=(12,12))
    plt.plot(x,y,'ko')
    ax=plt.gca()
    plt.axis('off')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.tight_layout()
    plt.axis('equal')
    plt.plot()
    plt.xlim([0,n+1])
    plt.ylim([0,n+1])
    ax=plt.gca()
    pats=[]
    for i in range(nbands):
        color=scheme_choice[i]
        ax.add_patch(pat.Rectangle((0,0.5+n/2+(round(nbands/2)-i)*spacing),n+1,1,facecolor=color))
    return f

def randpot1d_four(cf):
    n=cf.shape[0]
    p=(2*np.random.rand(n//2-1)-1)*pi
    phase=np.zeros(n)
    phase[1:n//2]=p
    phase[n//2+1:]=-np.fliplr([p])[0]
    phase[n//2]=0
    cff=cf*exp(1.j*phase)
    cff[0]=0
    
    u=sqrt(n)*ifft(cff)
    
    u=np.real(u)
    s=np.std(u)
    return u/s

def randpot1d_getFFT(c):
    f=fft(ifftshift(c))
    return complex(1,0)*sqrt(np.abs(f))


def InverseKick(R,cf,K):
    global dp
    # R is a (M,2) array of phase space coordinates
    # cf is fourier transform of the correlation of the kick
    # K is the kick strength
    
    X=np.array(R[:,0])
    dp=K*randpot1d_four(cf)
    Rp=np.array(R).copy()
    ix=np.floor(X).astype('int')
    Rp[:,1]=Rp[:,1]-dp[ix]-(dp[np.mod(ix+1,dp.shape[0])]-dp[ix])*(X-ix)
    return(Rp)
    
def InverseDrift(R,row0,Tau):
    # R is a (M,2) array of phase space coordinates
    # row0 is the row in the image that corresponds to zero momentum
    # Tau os the time of the drift period
    Rp=np.array(R).copy()
    Rp[:,0]=Rp[:,0]+(Rp[:,1]-row0)*Tau
    return(Rp)
    
def InverseDrift(R,row0,Tau):
    # R is a (M,2) array of phase space coordinates
    # row0 is the row in the image that corresponds to zero momentum
    # Tau os the time of the drift period
    Rp=np.array(R).copy()
    Rp[:,0]=Rp[:,0]+(Rp[:,1]-row0)*Tau
    return(Rp)

@app.route('/plt0.png')
def plt0():
    filename = 'iteration-0.png'
    return send_file(filename, mimetype='image/png')
    
@app.route('/plt1.png')
def plt1():
    filename = 'iteration-1.png'
    return send_file(filename, mimetype='image/png')

@app.route('/plt2.png')
def plt2():
    filename = 'iteration-2.png'
    return send_file(filename, mimetype='image/png')
@app.route('/plt3.png')
def plt3():
    filename = 'iteration-3.png'
    return send_file(filename, mimetype='image/png')
@app.route('/plt4.png')
def plt4():
    filename = 'iteration-4.png'
    return send_file(filename, mimetype='image/png')
@app.route('/plt5.png')
def plt5():
    filename = 'iteration-5.png'
    return send_file(filename, mimetype='image/png')

@app.route('/plt6.png')
def plt6():
    filename = 'iteration-6.png'
    return send_file(filename, mimetype='image/png')

def generate_plots(ka, dp):
    maxiteration=7
        
    K0=ka
    Tau0=dp

    rimage = imageio.imread("init_image.png")
    rows, cols = rimage.shape[0], rimage.shape[1]
    
    image=rimage
    if rows %2 != 0 :
        image=rimage[1:,:,:]
    if cols %2 != 0 :
        image=image[:,1:,:]
    rows, cols = image.shape[0], image.shape[1]
    
    # The center row that corresponds to momentum 0
    row0=rows/2.0
  
    #Kicks will be Gaussian correlated with correlation length lc
    lc=0.05
    
    #some preparations
    Lx=1.0
    Lp=1.0
    K=K0/np.float(Lp)*rows
    Tau=np.float(Tau0*Lx*rows)/(cols*Lp)
    #Create a Gaussian correlation
    x=np.linspace(-np.float(Lx)/2,np.float(Lx)/2,num=rows,endpoint=True)
    c=np.exp(-x**2/lc**2)
    cf=randpot1d_getFFT(c)
    plt.close('all')
    #figurelist=[]
    for iteration in range(0, maxiteration):
        str="th"
        if iteration < 2:
            str=["st","nd"][iteration]
        # print("doing iteration ",iteration+1," out of ",maxiteration)
        #Warp image with the kick part of the map, using the inverse mapping
        stage1 = warp(image, InverseKick,map_args={'cf': cf,'K': K},mode='wrap',order=1)
        #Warp image with the drift part of the map, using the inverse mapping
        stage2 = warp(stage1, InverseDrift,map_args={'row0': row0,'Tau': Tau},mode='wrap',order=1)
        f, (p1, p2, p3) = plt.subplots(1, 3,sharey=True,figsize=(24,8))
        f.subplots_adjust(wspace=0, hspace=0)
        #plt.tight_layout()
        p1.axis('equal')
        p1.axis('off')
        p2.axis('equal')
        p2.axis('off')
        p3.axis('equal')
        p3.axis('off')
        p1.imshow(image)
        if iteration==0:
            p1.set_title('initial condition')
        else:
            p1.set_title('after %d%s kick-drift cycle '%(iteration,str))
        p2.imshow(stage1)
        p2.set_title('after kick')
        p2.plot(rows/2+dp,color='c',linewidth=3)
        #p2.plot([0,cols],[rows/2,rows/2],":",linewidth=1)
        p3.imshow(stage2)
        p3.set_title('after %d%s kick-drift cycle '%(iteration+1,str))
        #output = io.BytesIO()
        #FigureCanvas(f).print_png(output)
        f.savefig("iteration-%d.png"%iteration, dpi=160, facecolor='w')
        #figurelist.append(f)
        image=stage2
    
    results=[]
    for iteration in range(0, maxiteration):
        img = imageio.imread("iteration-%d.png"%iteration)
        results.append(img)
    return(results)
        
        
if __name__ == '__main__':
    app.debug = True
    app.run()
