"""Native file selection and worker control; numerical work stays off the UI thread."""

from datetime import datetime
import json
from pathlib import Path
import queue
import shutil
import subprocess
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import uuid


class Desktop:
    def __init__(self, root):
        self.root = root
        self.process = None
        self.result = None
        self.servers = []
        self.events = queue.Queue()
        self.cancelled = False
        root.title('faceMap Desktop')
        root.geometry('680x550')
        root.minsize(590, 510)
        root.protocol('WM_DELETE_WINDOW', self.close)
        root.columnconfigure(0, weight=1)
        frame = ttk.Frame(root, padding=26)
        frame.grid(sticky='nsew')
        frame.columnconfigure(0, weight=1)
        ttk.Label(frame, text='Turn your capture into a 3D model', font=('', 21, 'bold'), wraplength=600).grid(sticky='w', pady=(0, 10))
        ttk.Label(frame, text='Export a video or scan from faceMap on your iPhone, then choose it here.\nYour capture is processed on this computer and is never uploaded.', wraplength=610).grid(sticky='w', pady=(0, 20))
        self.source = tk.StringVar()
        self.destination = tk.StringVar(value=str(Path.home() / 'Documents' / 'faceMap Results'))
        self.preset = tk.StringVar(value='balanced')
        self.mode = tk.StringVar(value='auto')
        self.status = tk.StringVar(value='Choose a video or scan to begin.')
        self.controls = []
        ttk.Label(frame, text='Video or scan ZIP').grid(sticky='w')
        row = ttk.Frame(frame)
        row.grid(sticky='ew', pady=(5, 14))
        row.columnconfigure(0, weight=1)
        field = ttk.Entry(row, textvariable=self.source)
        field.grid(row=0, column=0, sticky='ew')
        choose = ttk.Button(row, text='Choose capture...', command=self.choose)
        choose.grid(row=0, column=1, padx=(8, 0))
        self.controls.extend([field, choose])
        ttk.Label(frame, text='Save results in').grid(sticky='w')
        row = ttk.Frame(frame)
        row.grid(sticky='ew', pady=(5, 14))
        row.columnconfigure(0, weight=1)
        field = ttk.Entry(row, textvariable=self.destination)
        field.grid(row=0, column=0, sticky='ew')
        choose = ttk.Button(row, text='Choose folder...', command=self.choose_destination)
        choose.grid(row=0, column=1, padx=(8, 0))
        self.controls.extend([field, choose])
        row = ttk.Frame(frame)
        row.grid(sticky='ew', pady=(0, 12))
        ttk.Label(row, text='Processing').grid(row=0, column=0, sticky='w')
        quality = ttk.Combobox(row, textvariable=self.preset, values=['quick', 'balanced', 'detail'], state='readonly', width=14)
        quality.grid(row=1, column=0, padx=(0, 16), pady=(4, 0))
        ttk.Label(row, text='Capture data').grid(row=0, column=1, sticky='w')
        mode = ttk.Combobox(row, textvariable=self.mode, values=['auto', 'depth', 'photos'], state='readonly', width=14)
        mode.grid(row=1, column=1, pady=(4, 0))
        self.controls.extend([quality, mode])
        ttk.Label(frame, text='Automatic supports videos and older LiDAR scans. Video models have no measured scale.\nQuick uses less memory; detail takes longer. No dedicated GPU is required.', wraplength=610).grid(sticky='w', pady=(0, 14))
        self.bar = ttk.Progressbar(frame, maximum=100)
        self.bar.grid(sticky='ew', pady=(0, 8))
        ttk.Label(frame, textvariable=self.status, wraplength=600).grid(sticky='w', pady=(0, 14))
        row = ttk.Frame(frame)
        row.grid(sticky='w')
        self.start = ttk.Button(row, text='Create 3D model', command=self.run)
        self.start.grid(row=0, column=0)
        self.cancel = ttk.Button(row, text='Cancel', state='disabled', command=self.stop)
        self.cancel.grid(row=0, column=1, padx=8)
        self.view = ttk.Button(row, text='View model', state='disabled', command=self.show_result)
        self.view.grid(row=0, column=2)
        ttk.Button(frame, text='Open an existing result...', command=self.open_result).grid(sticky='w', pady=(15, 0))
        root.after(150, self.poll)

    def choose(self):
        path = filedialog.askopenfilename(title='Choose a faceMap export', filetypes=[
            ('Video or faceMap scan', '*.mov *.MOV *.mp4 *.MP4 *.m4v *.M4V *.zip'), ('All files', '*')])
        if path:
            self.source.set(path)

    def choose_destination(self):
        path = filedialog.askdirectory(title='Choose a folder for results')
        if path:
            self.destination.set(path)

    def run(self):
        if self.process:
            return
        if not Path(self.source.get()).is_file():
            messagebox.showerror('Choose a capture', 'Select a video or scan ZIP exported from faceMap first.')
            return
        self.result = Path(self.destination.get()).expanduser() / ('Scan-' + datetime.now().strftime('%Y%m%d-%H%M%S') + '-' + uuid.uuid4().hex[:6])
        entry = [sys.executable, '--worker'] if getattr(sys, 'frozen', False) else [sys.executable, '-m', 'facemap_desktop']
        command = [*entry, self.source.get(), '--output', str(self.result),
                   '--preset', self.preset.get(), '--mode', self.mode.get(), '--json-progress']
        try:
            self.process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding='utf-8', errors='replace', creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        except OSError as error:
            messagebox.showerror('Could not start processing', str(error))
            return
        self.cancelled = False
        self.last_error = None
        self.bar['value'] = 0
        self.status.set('Starting the CPU reconstruction engine...')
        self.start['state'], self.view['state'], self.cancel['state'] = 'disabled', 'disabled', 'normal'
        for item in self.controls:
            item['state'] = 'disabled'
        threading.Thread(target=self.read_worker, args=(self.process,), daemon=True).start()

    def read_worker(self, process):
        for line in process.stdout:
            try:
                self.events.put(json.loads(line))
            except json.JSONDecodeError:
                pass
        self.events.put({'exit_code': process.wait()})

    def poll(self):
        while not self.events.empty():
            event = self.events.get()
            if 'percent' in event:
                self.bar['value'] = event['percent']
                self.status.set(event['message'])
            if 'error' in event:
                self.last_error = event['error']
            if 'exit_code' in event:
                self.process = None
                self.start['state'], self.cancel['state'] = 'normal', 'disabled'
                for item in self.controls:
                    item['state'] = 'readonly' if isinstance(item, ttk.Combobox) else 'normal'
                report = {}
                if self.result and (self.result / 'report.json').is_file():
                    report = json.loads((self.result / 'report.json').read_text(encoding='utf-8'))
                if self.cancelled:
                    # A killed worker cannot execute its finally blocks. Clean only its
                    # private import snapshots; keep the input and failed report.
                    if self.result and self.result.is_dir():
                        for path in self.result.glob('facemap-import-*'):
                            if path.is_dir():
                                shutil.rmtree(path)
                        for name in ('model.glb', 'surface-mm.ply', 'surface.ply', 'report.json.tmp'):
                            (self.result / name).unlink(missing_ok=True)
                        report.update(status='cancelled')
                        (self.result / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
                    self.status.set('Cancelled. Your original scan is unchanged.')
                elif event['exit_code'] == 0 and report.get('status') == 'complete':
                    self.status.set('Model saved. Open the viewer to inspect the surface and any gaps.')
                    self.view['state'] = 'normal'
                else:
                    self.status.set(self.last_error or report.get('error') or 'Processing stopped. Try Quick mode or check the result report.')
        self.root.after(150, self.poll)

    def stop(self):
        if self.process and self.process.poll() is None:
            self.cancelled = True
            self.cancel['state'] = 'disabled'
            self.status.set('Cancelling...')
            self.process.terminate()

    def show_result(self):
        try:
            from .viewer import serve_viewer
            server, _ = serve_viewer(self.result)
            self.servers.append(server)
        except Exception as error:
            messagebox.showerror('Could not open model', str(error))

    def open_result(self):
        if self.process:
            return
        path = filedialog.askdirectory(title='Choose a completed faceMap result folder')
        if path:
            self.result = Path(path)
            self.show_result()

    def close(self):
        if self.process:
            self.stop()
            self.root.after(200, self.close)
            return
        for server in self.servers:
            server.shutdown()
            server.server_close()
        self.root.destroy()


def launch():
    root = tk.Tk()
    Desktop(root)
    root.mainloop()
