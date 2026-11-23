"""A mathematical 3D animation of non-identical returns.
Lorenz flow and a z=25 Poincare section; deterministic, not a proof of Deleuze.
Requirements: numpy, scipy, opencv-python, Pillow, ffmpeg.
"""
from __future__ import annotations
import os, subprocess
from pathlib import Path
import numpy as np
import cv2
from scipy.integrate import solve_ivp
from PIL import Image, ImageDraw, ImageFont

ROOT=Path('/mnt/data')
OUT=ROOT/'difference_returns_3d.mp4'
POSTER=ROOT/'difference_returns_3d.jpg'
W,H,FPS,FRAMES=1280,720,24,360
BG=np.array([16, 13, 10],dtype=np.uint8) # BGR
FONT='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
FONT_B='/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc'

def font(sz,bold=False):
    return ImageFont.truetype(FONT_B if bold else FONT,sz)

# Integrate one autonomous, deterministic vector field: no randomness.
sigma,rho,beta=10.0,28.0,8/3
DT=.012
T=110
sol=solve_ivp(lambda t,q:[sigma*(q[1]-q[0]),q[0]*(rho-q[2])-q[1],q[0]*q[1]-beta*q[2]],(0,T), [0.1,1,1.05], t_eval=np.arange(0,T,DT), rtol=4e-8, atol=1e-10)
P=sol.y.T
TIME=sol.t
# Cross a fixed section (z=25, upward), on the right lobe (x>0)
inds=np.where((P[:-1,2]<25)&(P[1:,2]>=25)&(P[:-1,0]>0))[0]
f=(25-P[inds,2])/(P[inds+1,2]-P[inds,2]); C=P[inds]+f[:,None]*(P[inds+1]-P[inds])
Tc=TIME[inds]+f*DT
P0=np.array([0,0,25.])
# Crop the cross-section view to the same observed right-lobe trajectories
s=max(15,float(Tc[0])); end=94

def projection(coords, u):
    """Project actual XYZ onto a moving orthographic camera; no perspective guides."""
    az=np.deg2rad(-47+33*np.sin(u*1.42-0.1))
    el=np.deg2rad(23+10*np.cos(u*1.05))
    right=np.array([-np.sin(az), np.cos(az),0])
    up=np.array([-np.sin(el)*np.cos(az), -np.sin(el)*np.sin(az), np.cos(el)])
    depth=np.array([np.cos(el)*np.cos(az),np.cos(el)*np.sin(az),np.sin(el)])
    c=np.array([0.,0.,25.])
    v=coords-c
    x=520+10.2*(v@right)
    y=397-10.2*(v@up)
    d=v@depth
    return np.c_[x,y],d

def poly(img, pp, color, thickness=1, closed=False):
    cv2.polylines(img, [np.round(pp).astype(np.int32).reshape(-1,1,2)],closed, color, thickness, cv2.LINE_AA)

def add_text(frame,xy,content,fill,size,bold=False):
    # PIL text is rendered into the BGR canvas (converted via RGB)
    img=Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    d=ImageDraw.Draw(img)
    d.text(xy,content,font=font(size,bold),fill=fill)
    return cv2.cvtColor(np.asarray(img),cv2.COLOR_RGB2BGR)

# Static labels rendered once as transparent overlay
ol=Image.new('RGBA',(W,H),(0,0,0,0)); dr=ImageDraw.Draw(ol)
dr.text((64,47),'DIFFERENCE RETURNS',font=font(29,True),fill=(229,238,241,245))
dr.text((65,94),'差異的回返',font=font(20),fill=(152,183,195,230))
dr.text((65,128),'同一生成規則  ·  不同回返軌跡',font=font(15),fill=(122,153,168,210))
dr.line((929,67,929,654),fill=(67,82,96,170),width=1)
dr.text((968,52),'RECURSION  /  RETURN',font=font(18,True),fill=(225,235,239,240))
dr.text((967,93),'相同截面  z = 25',font=font(17),fill=(153,193,201,228))
dr.text((968,121),'每次經過都留下不同位置',font=font(14),fill=(115,147,163,216))
dr.text((967,504),'同一套微分方程',font=font(17,True),fill=(200,219,224,240))
dr.text((967,535),'dx/dt = 10(y − x)',font=font(15),fill=(150,176,188,230))
dr.text((967,561),'dy/dt = x(28 − z) − y',font=font(15),fill=(150,176,188,230))
dr.text((967,587),'dz/dt = xy − 8z/3',font=font(15),fill=(150,176,188,230))
dr.text((64,670),'Lorenz flow  ·  deterministic, non-periodic trajectory  ·  3D phase space',font=font(13),fill=(132,149,162,216))
OVERLAY=np.asarray(ol).astype(np.uint16)

def compose_overlay(frame):
    # RGBA to BGR output
    a=(OVERLAY[:,:,3:4]/255.0)
    rgb=OVERLAY[:,:,:3][:,:,::-1]
    return np.clip(frame*(1-a)+rgb*a,0,255).astype(np.uint8)

def draw_frame(i):
    u=i/(FRAMES-1)
    tt=s+(end-s)*u
    cur=min(int(tt/DT),len(P)-1)
    frame=np.empty((H,W,3),dtype=np.uint8);frame[:]=BG
    # very faint gradient and fine stars (not a perspective grid)
    for yy in range(H):
        frame[yy,:,0]=np.uint8(13+int(yy/H*13))
        frame[yy,:,1]=np.uint8(14+int(yy/H*4))
        frame[yy,:,2]=np.uint8(18+int(yy/H*2))
    # 3D section plane, only an actual plane, not decorative perspective lines
    plane=np.array([[0,-23,25],[24,-23,25],[24,23,25],[0,23,25]],dtype=float)
    section,_=projection(plane,u)
    canvas=frame.copy()
    cv2.fillConvexPoly(canvas,np.round(section).astype(np.int32),(70,55,32),cv2.LINE_AA)
    cv2.addWeighted(canvas,.13,frame,.87,0,frame)
    poly(frame,section,(91,86,61),1,True)
    # faint orbit substrate (whole shape serves as phase-space context)
    ghost, gd=projection(P[1350::6],u)
    poly(frame,ghost,(51,54,48),1)
    # trail in age bands, projected from actual 3D coordinates
    hist=4500
    lo=max(0,cur-hist)
    traj=P[lo:cur+1]
    xy,depth=projection(traj,u)
    N=len(xy)
    if N>10:
        # Older paths: cold and pale; fresh path: brighter and warmer.
        bands=[(0,.38,(81,74,47),1),(.38,.70,(125,126,60),1),(.70,.90,(178,166,78),2),(.90,.97,(189,204,114),2),(.97,1,(219,240,169),3)]
        for a,b,c,w in bands:
            j0=int((N-1)*a);j1=max(j0+2,int((N-1)*b))
            poly(frame,xy[j0:j1],c,w)
        # new glow + particle
        xy_end=tuple(np.rint(xy[-1]).astype(int))
        glow=frame.copy()
        cv2.circle(glow,xy_end,16,(105,178,139),-1,cv2.LINE_AA)
        cv2.GaussianBlur(glow,(0,0),10,dst=glow)
        cv2.addWeighted(frame,.92,glow,.08,0,frame)
        cv2.circle(frame,xy_end,8,(169,211,191),1,cv2.LINE_AA)
        cv2.circle(frame,xy_end,4,(232,250,249),-1,cv2.LINE_AA)
    # Plot return positions in 3D on the actual Poincare section
    past=Tc<=tt; pts=C[past]
    if len(pts):
        pc, dep=projection(pts,u)
        for k,p in enumerate(pc):
            col=(109,137,123) if k<len(pc)-1 else (212,246,211)
            cv2.circle(frame,tuple(np.rint(p).astype(int)),3 if k<len(pc)-1 else 5,col,-1,cv2.LINE_AA)
    # Right inset: section coordinates, no arbitrary perspective grid
    x0,y0=968,184
    x1,y1=1205,451
    cv2.rectangle(frame,(x0,y0),(x1,y1),(27,31,32),-1)
    cv2.rectangle(frame,(x0,y0),(x1,y1),(62,74,80),1)
    # coordinate axes, only math axes
    cv2.line(frame,(x0+24,y1-23),(x1-17,y1-23),(66,92,96),1,cv2.LINE_AA)
    cv2.line(frame,(x0+24,y1-23),(x0+24,y0+17),(66,92,96),1,cv2.LINE_AA)
    # x and y of section points; project to inset without moving limits
    def px(a):return np.array([x0+24+(a[:,0]-0)/27*(x1-x0-46),y1-23-(a[:,1]+24)/49*(y1-y0-42)]).T
    pp=px(pts) if len(pts) else np.empty((0,2))
    if len(pp):
        for j,p in enumerate(pp):
            c=(89,143,139) if j<len(pp)-1 else (174,229,206)
            cv2.circle(frame,tuple(np.rint(p).astype(int)),3 if j<len(pp)-1 else 6,c,-1,cv2.LINE_AA)
        if len(pp)>1:
            for j in range(1,len(pp)):
                cv2.line(frame,tuple(np.rint(pp[j-1]).astype(int)),tuple(np.rint(pp[j]).astype(int)),(53,79,73),1,cv2.LINE_AA)
    frame=compose_overlay(frame)
    # Current count
    return_count=len(pts)
    frame=add_text(frame,(969,460),f'RETURNS   {return_count:02d}',(213,222,226),16,True)
    return frame

if __name__=='__main__':
    # Stream raw BGR directly into ffmpeg, no temporary frame folder.
    cmd=['ffmpeg','-hide_banner','-loglevel','error','-y','-f','rawvideo','-vcodec','rawvideo','-pix_fmt','bgr24',
        '-s',f'{W}x{H}','-r',str(FPS),'-i','-', '-an','-c:v','libx264','-preset','veryfast','-crf','19','-pix_fmt','yuv420p','-movflags','+faststart',str(OUT)]
    p=subprocess.Popen(cmd,stdin=subprocess.PIPE)
    assert p.stdin
    for k in range(FRAMES):
        img=draw_frame(k)
        p.stdin.write(img.tobytes())
        if k==int(FRAMES*.77):
            cv2.imwrite(str(POSTER),img,[cv2.IMWRITE_JPEG_QUALITY,93])
        if k%80==0:print(f'{k}/{FRAMES}',flush=True)
    p.stdin.close();ret=p.wait()
    if ret:raise RuntimeError(f'ffmpeg exit {ret}')
    print('Wrote',OUT,OUT.stat().st_size,'poster',POSTER.stat().st_size)