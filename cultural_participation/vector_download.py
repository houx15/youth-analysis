"""下载公开静态词向量，支持HTTP断点；Google Drive仅处理公开文件确认表单。"""
import argparse
from html.parser import HTMLParser
from pathlib import Path
import time

import requests


class DownloadForm(HTMLParser):
    def __init__(self):
        super().__init__(); self.action=None; self.fields={}
    def handle_starttag(self,tag,attrs):
        attrs=dict(attrs)
        if tag=='form' and attrs.get('id')=='download-form': self.action=attrs.get('action')
        if tag=='input' and attrs.get('name'): self.fields[attrs['name']]=attrs.get('value','')


def download(url,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    partial=path.with_suffix(path.suffix+'.part')
    if path.exists():
        print(f'已存在：{path}',flush=True);return
    with requests.Session() as session:
        if 'drive.usercontent.google.com' in url:
            with session.get(url,stream=True,timeout=60) as response:
                response.raise_for_status()
                if 'text/html' in response.headers.get('Content-Type',''):
                    form=DownloadForm();form.feed(response.text)
                    if form.action!='https://drive.usercontent.google.com/download':
                        raise ValueError('未识别公开下载确认表单')
                    url=requests.Request('GET',form.action,params=form.fields).prepare().url
        for attempt in range(4):
            size=partial.stat().st_size if partial.exists() else 0
            try:
                headers={'Range':f'bytes={size}-'} if size else {}
                with session.get(url,headers=headers,stream=True,timeout=(30,120)) as response:
                    response.raise_for_status()
                    if size and (response.status_code!=206 or not response.headers.get('Content-Range','').startswith(f'bytes {size}-')):
                        raise ValueError('服务器未接受断点，保留临时文件，不追加错误内容')
                    if 'html' in response.headers.get('Content-Type',''): raise ValueError('返回HTML而非向量压缩包')
                    expected=int(response.headers.get('Content-Length',0));written=0;last=time.monotonic()
                    with partial.open('ab' if size else 'wb') as stream:
                        for block in response.iter_content(1024*1024):
                            stream.write(block);written+=len(block)
                            if time.monotonic()-last>20:
                                print(f'{path.name}: {(size+written)/1e6:.0f} MB',flush=True);last=time.monotonic()
                    if expected and written!=expected: raise IOError('文件长度不符')
                partial.rename(path);print(f'下载完成：{path} ({path.stat().st_size:,} bytes)',flush=True);return
            except requests.RequestException:
                if attempt==3: raise
                print('网络中断，尝试从已下载位置继续',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('url');p.add_argument('output');a=p.parse_args();download(a.url,a.output)
