import qdarkstyle
import warnings
import torch, os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

warnings.simplefilter(action='ignore', category=FutureWarning)
from transformers import CLIPTextModel, CLIPTokenizer

from diffusers import AutoencoderKL, DDIMScheduler

from src.unet_3d import MyUNet3DConditionModel as UNet3DConditionModel
from src.t2v_pipeline import TextToVideoSDPipeline

from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QLabel, QVBoxLayout,QGroupBox,  QHBoxLayout, QPushButton


class MainUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Direct-a-Video User Interface')
        # self.setGeometry(100, 100, 400, 600)

        # Set global font size
        font = QFont()
        font.setPointSize(12)  # Set the font size to 12
        QApplication.setFont(font)

        mainWidget = QWidget(self)
        self.setCentralWidget(mainWidget)
        layout_main = QHBoxLayout(mainWidget)

        section_txt = QGroupBox("Text Prompt")
        layout_txt = QVBoxLayout(section_txt)

        section_cam = QGroupBox("Camera Movement")
        layout_cam = QVBoxLayout(section_cam)

        section_obj = QGroupBox("Object Motion")
        layout_obj = QVBoxLayout(section_obj)

        section_extra = QGroupBox("Extra Settings")
        layout_extra = QVBoxLayout(section_extra)

        section_out = QGroupBox("Output")
        layout_out = QVBoxLayout(section_out)

        layout_te = QVBoxLayout()
        layout_te.addWidget(section_txt)
        layout_te.addWidget(section_extra)


        layout_cov = QVBoxLayout()
        layout_co = QHBoxLayout()
        layout_co.addWidget(section_cam)
        layout_co.addWidget(section_obj)
        layout_cov.addLayout(layout_co)
        layout_cov.addWidget(section_out)

        layout_main.addLayout(layout_te)
        layout_main.addLayout(layout_cov)


        def __txt__(): pass
        from section_txt import TxtSection
        self.txtSection = TxtSection()

        self.txtSection.layout_cfg = QHBoxLayout()
        self.txtSection.layout_cfg.addWidget(QLabel("CFG scale:"))
        self.txtSection.layout_cfg.addWidget(self.txtSection.slider_cfg)
        self.txtSection.layout_cfg.addWidget(self.txtSection.value_cfg)

        # 将组件添加到main布局
        # layout_txt.addWidget(self.txtSection.enableSwitch)
        layout_txt.addWidget(self.txtSection.textPromptLabel)
        layout_txt.addWidget(self.txtSection.textPrompt)
        layout_txt.addWidget(self.txtSection.NegPromptLabel)
        layout_txt.addWidget(self.txtSection.negPrompt)
        layout_txt.addLayout(self.txtSection.layout_cfg)
        layout_txt.addWidget(self.txtSection.textResetButton)


        def __cam__(): pass
        from section_cam import CamSection
        self.camSection = CamSection()

        self.camSection.layout_cam_x = QHBoxLayout()
        self.camSection.layout_cam_x.addWidget(QLabel("X-pan ratio:"))
        self.camSection.layout_cam_x.addWidget(self.camSection.slider_cam_x)
        self.camSection.layout_cam_x.addWidget(self.camSection.value_cam_x)

        self.camSection.layout_cam_y = QHBoxLayout()
        self.camSection.layout_cam_y.addWidget(QLabel("Y-pan ratio:"))
        self.camSection.layout_cam_y.addWidget(self.camSection.slider_cam_y)
        self.camSection.layout_cam_y.addWidget(self.camSection.value_cam_y)

        self.camSection.layout_cam_z = QHBoxLayout()
        self.camSection.layout_cam_z.addWidget(QLabel("Zoom ratio:"))
        self.camSection.layout_cam_z.addWidget(self.camSection.slider_cam_z)
        self.camSection.layout_cam_z.addWidget(self.camSection.value_cam_z)

        self.camSection.layout_cam_off = QHBoxLayout()
        self.camSection.layout_cam_off.addWidget(QLabel("Camera off timestep:"))
        self.camSection.layout_cam_off.addWidget(self.camSection.slider_cam_off)
        self.camSection.layout_cam_off.addWidget(self.camSection.value_cam_off)

        layout_cam.addWidget(self.camSection.enableSwitch)
        layout_cam.addLayout(self.camSection.layout_cam_x)
        layout_cam.addLayout(self.camSection.layout_cam_y)
        layout_cam.addLayout(self.camSection.layout_cam_z)
        layout_cam.addWidget(self.camSection.camCfgCheckbox)
        layout_cam.addLayout(self.camSection.layout_cam_off)
        layout_cam.addWidget(self.camSection.resetCamButton)


        def __obj__(): pass
        from section_obj import ObjSection

        self.objSection = ObjSection()
        layout_obj.addWidget(self.objSection.enableSwitch)
        layout_obj.addWidget(self.objSection)

        self.objSection.layout_btn = QHBoxLayout()
        self.objSection.layout_btn.addWidget(self.objSection.addObjBtn)
        self.objSection.layout_btn.addWidget(self.objSection.finishBtn)
        self.objSection.layout_btn.addWidget(self.objSection.clearAllBtn)
        layout_obj.addLayout(self.objSection.layout_btn)
        layout_obj.addLayout(self.objSection.layout_tau)
        layout_obj.addLayout(self.objSection.layout_creg)
        layout_obj.addWidget(self.objSection.saveButton)


        def __output__():pass
        from section_videoPlayer import VideoPlayer
        self.videoPlayer = VideoPlayer()

        self.videoPlayer.layout_btn = QHBoxLayout()
        self.videoPlayer.layout_btn.addWidget(self.videoPlayer.pauseButton)
        self.videoPlayer.layout_btn.addWidget(self.videoPlayer.clearButton)
        self.videoPlayer.layout_btn.addWidget(self.videoPlayer.saveButton)

        layout_out.addWidget(self.videoPlayer.videoLabel)
        layout_out.addLayout(self.videoPlayer.layout_btn)
        layout_out.addWidget(self.videoPlayer.generateButton)

        # # 开始时隐藏运行按钮
        self.videoPlayer.generateButton.hide()
        self.videoPlayer.clearButton.hide()
        self.videoPlayer.saveButton.hide()
        self.videoPlayer.pauseButton.hide()

        self.videoPlayer.generateButton.clicked.connect(self.run)

        def __init__():pass
        self.initButton = QPushButton(">>> Click here to initialize the model first! <<<")
        layout_out.addWidget(self.initButton)
        self.initButton.clicked.connect(self.init_model)

        def __extra__():pass
        from section_extra import ExtraSection
        self.extraSection = ExtraSection()

        layout_extra.addLayout(self.extraSection.layout_h)
        layout_extra.addLayout(self.extraSection.layout_w)
        layout_extra.addLayout(self.extraSection.layout_f)
        layout_extra.addLayout(self.extraSection.layout_seed)
        layout_extra.addLayout(self.extraSection.layout_step)


    def init_model(self):
        self.initButton.setEnabled(False)
        self.initButton.setText("Initializing, please wait... ")
        QApplication.processEvents()  # 允许Qt处理其他事件

        self.device = torch.device("cuda:0")
        self.dtype = torch.float16  # recommended to use float16 for faster inference

        pretrained_model_name_or_path = "cerspense/zeroscope_v2_576w"  # original model path
        cam_ckpt_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),"ckpt/unet_cam_model.ckpt") # trained camera model path

        ## load unet
        unet_orig = UNet3DConditionModel.from_pretrained(pretrained_model_name_or_path, subfolder="unet",
                                                         torch_dtype=self.dtype)
        unet_config = UNet3DConditionModel.load_config(pretrained_model_name_or_path, subfolder='unet')
        unet_config['attention_type'] = 'cross_temp'
        unet = UNet3DConditionModel.from_config(unet_config)

        unet_orig_ckpt = unet_orig.state_dict()
        unet_cam_ckpt = torch.load(cam_ckpt_path, map_location='cpu')

        unet.load_state_dict({**unet_orig_ckpt, **unet_cam_ckpt}, strict=True)
        unet.to(dtype=self.dtype)
        del unet_orig, unet_cam_ckpt

        unet.set_direct_a_video_attn_processors()

        ## load other models
        vae = AutoencoderKL.from_pretrained(pretrained_model_name_or_path, subfolder="vae", torch_dtype=self.dtype)
        vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
        tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_name_or_path, subfolder='tokenizer',
                                                  torch_dtype=self.dtype, )
        text_encoder = CLIPTextModel.from_pretrained(pretrained_model_name_or_path, subfolder='text_encoder',
                                                     torch_dtype=self.dtype, )
        scheduler = DDIMScheduler.from_pretrained(pretrained_model_name_or_path, subfolder='scheduler',
                                                  torch_dtype=self.dtype)
        # scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
        ## make pipeline
        self.pipeline = TextToVideoSDPipeline(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
        ).to(self.device)
        self.pipeline.set_progress_bar_config(disable=False)

        # # show buttons after init
        self.videoPlayer.generateButton.show()
        self.videoPlayer.clearButton.show()
        self.videoPlayer.saveButton.show()
        self.videoPlayer.pauseButton.show()
        self.initButton.hide()
        print("Loading completed!")

    def run(self):
        h = int(self.extraSection.height_group.button(self.extraSection.height_group.checkedId()).text())
        w = int(self.extraSection.width_group.button(self.extraSection.width_group.checkedId()).text())
        f = int(self.extraSection.frame_group.button(self.extraSection.frame_group.checkedId()).text())
        num_inference_steps = int(self.extraSection.slider_step.value())

        self.objSection.finishDrawing(num_boxes=f)
        self.videoPlayer.pauseButton.setEnabled(False)
        self.videoPlayer.clearButton.setEnabled(False)
        self.videoPlayer.saveButton.setEnabled(False)
        self.videoPlayer.generateButton.setEnabled(False)
        self.initButton.setEnabled(False)

        prompt = self.txtSection.textPrompt.toPlainText()
        neg_prompt = self.txtSection.negPrompt.toPlainText()
        cfg = self.txtSection.slider_cfg.value()

        cam_availability = self.camSection.enableSwitch.isChecked()
        cam_x = self.camSection.slider_cam_x.value() / 100
        cam_y = self.camSection.slider_cam_y.value() / 100
        cam_z = 2 ** (self.camSection.slider_cam_z.value() / 100)
        cam_cfg = self.camSection.camCfgCheckbox.isChecked()
        cam_off_t = self.camSection.slider_cam_off.value() / 100

        obj_availability = self.objSection.enableSwitch.isChecked()
        bbox = self.objSection.box_matrix
        bbox = [torch.from_numpy(b) for b in bbox]
        attn_lambda = self.objSection.slider_creg.value()
        attn_tau = self.objSection.slider_tau.value() / 100

        if self.extraSection.radio_fixed.isChecked():
            seed = int(self.extraSection.seed_input.text())
            generator = torch.Generator(device=self.device).manual_seed(seed)
        else:
            generator = None
        ########################################
        if cam_availability:
            cam_motion = torch.tensor([[[float(cam_x), float(cam_y), float(cam_z)]]]).to(device=self.device,
                                                                                         dtype=self.dtype)
        else:
            cam_motion = None
            cam_off_t = None
            cam_cfg = None

        ########################################
        if obj_availability:

            obj_motion_attn_kwargs = self.pipeline.prepare_obj_motion_attn_kwargs(prompt=prompt, bbox=bbox,
                                                                             attn_lambda=attn_lambda, attn_tau=attn_tau,
                                                                             h=h, w=w, f=f)

        else:
            obj_motion_attn_kwargs = None
        ###################################################

        def get_time_step(i, t_, t, latents, noise_pred):   # for progress bar use
            self.videoPlayer.drawProgressBar(i / num_inference_steps)
            QApplication.processEvents()  # 允许Qt处理其他事件

        out_np_list = self.pipeline(
            prompt=prompt,
            negative_prompt=neg_prompt,
            callback=get_time_step,
            cam_motion=cam_motion,
            cam_off_t=cam_off_t,  # time step to turn off camera motion control
            cam_cfg=cam_cfg,  # enable classifier-free guidance for camera motion control
            cross_attention_kwargs=obj_motion_attn_kwargs,
            num_frames=f,
            guidance_scale=cfg,
            num_inference_steps=num_inference_steps,
            width=w,
            height=h,
            generator=generator,
            output_type="np",
        ).frames
        # save_name = os.path.join(self.output_dir, f"__output__.mp4")
        # save_tensor_as_mp4(out, save_name)


        self.videoPlayer.playVideo(out_np_list)

        self.videoPlayer.pauseButton.setEnabled(True)
        self.videoPlayer.clearButton.setEnabled(True)
        self.videoPlayer.saveButton.setEnabled(True)
        self.videoPlayer.generateButton.setEnabled(True)
        self.initButton.setEnabled(True)



if __name__ == '__main__':
    app = QApplication(sys.argv)
    # app.setStyle("Fusion")

    # dark_theme_style = """
    # QWidget {
    #     background-color: #333333;
    #     color: #FFFFFF;
    # }
    # QPushButton {
    #     background-color: #555555;
    #     border: none;
    #     color: #FFFFFF;
    #     padding: 10px 20px;
    #     text-align: center;
    #     text-decoration: none;
    #     display: inline-block;
    #     font-size: 20px;
    #     border-radius: 5px;
    # }
    # QPushButton:hover {
    #     background-color: #777777;
    # }
    # """
    #
    # app.setStyleSheet(dark_theme_style)
    app.setStyleSheet(qdarkstyle.load_stylesheet())
    ex = MainUI()
    ex.show()
    sys.exit(app.exec_())