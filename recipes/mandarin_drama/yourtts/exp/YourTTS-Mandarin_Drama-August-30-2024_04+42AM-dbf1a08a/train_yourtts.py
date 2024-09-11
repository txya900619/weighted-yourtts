import os

import torch
from trainer import Trainer, TrainerArgs

from TTS.bin.compute_embeddings import compute_embeddings
from TTS.bin.resample import resample_files
from TTS.config.shared_configs import BaseDatasetConfig
from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.vits import CharactersConfig, Vits, VitsArgs, VitsAudioConfig
from TTS.utils.downloaders import download_vctk

torch.set_num_threads(24)

# pylint: disable=W0105
"""
    This recipe replicates the first experiment proposed in the YourTTS paper (https://arxiv.org/abs/2112.02418).
    YourTTS model is based on the VITS model however it uses external speaker embeddings extracted from a pre-trained speaker encoder and has small architecture changes.
    In addition, YourTTS can be trained in multilingual data, however, this recipe replicates the single language training using the VCTK dataset.
    If you are interested in multilingual training, we have commented on parameters on the VitsArgs class instance that should be enabled for multilingual training.
    In addition, you will need to add the extra datasets following the VCTK as an example.
"""
CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))

# Name of the run for the Trainer
RUN_NAME = "YourTTS-Mandarin_Drama"

# Path where you want to save the models outputs (configs, checkpoints and tensorboard logs)
OUT_PATH = os.path.dirname(os.path.abspath(__file__))  # "/raid/coqui/Checkpoints/original-YourTTS/"

# If you want to do transfer learning and speedup your training you can set here the path to the original YourTTS model
RESTORE_PATH = None  # "/root/.local/share/tts/tts_models--multilingual--multi-dataset--your_tts/model_file.pth"

# This paramter is useful to debug, it skips the training epochs and just do the evaluation  and produce the test sentences
SKIP_TRAIN_EPOCH = False

# Set here the batch size to be used in training and evaluation
BATCH_SIZE = 16

# Training Sampling rate and the target sampling rate for resampling the downloaded dataset (Note: If you change this you might need to redownload the dataset !!)
# Note: If you add new datasets, please make sure that the dataset sampling rate and this parameter are matching, otherwise resample your audios
SAMPLE_RATE = 16000

# Max audio length in seconds to be used in training (every audio bigger than it will be ignored)
MAX_AUDIO_LEN_IN_SECONDS = 15

### Download VCTK dataset
# VCTK_DOWNLOAD_PATH = os.path.join(CURRENT_PATH, "VCTK")
MD_DOWNLOAD_PATH = '/mnt/md1/user_wago/data/mandarin_drama'
# Define the number of threads used during the audio resampling
NUM_RESAMPLE_THREADS = 10
# Check if VCTK dataset is not already downloaded, if not download it
if not os.path.exists(MD_DOWNLOAD_PATH):
    raise ("MANDARIN_DRAMA dataset not exist")
    

# init configs
md_config = BaseDatasetConfig(
    formatter="mandarin_drama",
    dataset_name="mandarin_drama",
    meta_file_train="combined_mandarin_drama_text.csv",
    meta_file_val="",
    path=MD_DOWNLOAD_PATH,
    language="ma",
)

# Add here all datasets configs, in our case we just want to train with the VCTK dataset then we need to add just VCTK. Note: If you want to add new datasets, just add them here and it will automatically compute the speaker embeddings (d-vectors) for this new dataset :)
DATASETS_CONFIG_LIST = [md_config]

### Extract speaker embeddings
# SPEAKER_ENCODER_CHECKPOINT_PATH = (
#     "https://github.com/coqui-ai/TTS/releases/download/speaker_encoder_model/model_se.pth.tar"
# )
# SPEAKER_ENCODER_CONFIG_PATH = "https://github.com/coqui-ai/TTS/releases/download/speaker_encoder_model/config_se.json"

D_VECTOR_FILES = ['/mnt/md1/user_wago/data/mandarin_drama/md_speaker.json']  # List of speaker embeddings/d-vectors to be used during the training

# Audio config used in training.
audio_config = VitsAudioConfig(
    sample_rate=SAMPLE_RATE,
    hop_length=256,
    win_length=1024,
    fft_size=1024,
    mel_fmin=0.0,
    mel_fmax=None,
    num_mels=80,
)

# Init VITSArgs setting the arguments that are needed for the YourTTS model
model_args = VitsArgs(
    d_vector_file=D_VECTOR_FILES,
    use_d_vector_file=True,
    d_vector_dim=512,
    num_layers_text_encoder=10,
    resblock_type_decoder="2",  # In the paper, we accidentally trained the YourTTS using ResNet blocks type 2, if you like you can use the ResNet blocks type 1 like the VITS model
    # Useful parameters to enable the Speaker Consistency Loss (SCL) described in the paper
    # use_speaker_encoder_as_loss=True,
    # Useful parameters to enable multilingual training
    # use_language_embedding=True,
    # embedded_language_dim=4,
)

# General training config, here you can change the batch size and others useful parameters
config = VitsConfig(
    output_path=os.path.join(OUT_PATH,'exp'),
    model_args=model_args,
    run_name=RUN_NAME,
    project_name="YourTTS",
    run_description="""
            - Original YourTTS trained using VCTK dataset
        """,
    dashboard_logger="wandb",
    logger_uri=None,
    audio=audio_config,
    batch_size=BATCH_SIZE,
    batch_group_size=48,
    eval_batch_size=BATCH_SIZE,
    num_loader_workers=8,
    eval_split_max_size=256,
    print_step=50,
    plot_step=100,
    log_model_step=10000,
    save_step=5000,
    save_n_checkpoints=2,
    save_checkpoints=True,
    target_loss="loss_1",
    print_eval=False,
    use_phonemes=False,
    phonemizer="espeak",
    phoneme_language="ma",
    compute_input_seq_cache=True,
    add_blank=True,
    text_cleaner="chinese_mandarin_cleaners",
    characters=CharactersConfig(
        characters_class="TTS.tts.models.vits.VitsCharacters",
        pad="_",
        eos="&",
        bos="*",
        blank=None,
        characters=' \"%\'()*+-./0123456789?ABCDEFGHIJKLMNOPQRSTUVWXY\abcdefghiklmnoprstuvwxyz×ˇˊγ’…╳○、「」ㄅㄆㄇㄈㄌㄍㄎㄏㄐㄑㄒㄓㄕㄝㄟㄢㄣㄤㄥㄧㄨ一丁七丈三上下不丐且丕世丘丙丟並中串丸丹主乃久么之乍乎乏乒乖乘乙九乞也乩乳乾亂了予事二于云互五井些亞亡亢交亦亨享京亮人什仁仃仄仇今介仍仔他仗付仙代令以仰仲件任份仿企伊伍伎伏伐休伕伙伯估伴伶伸伺似伽但佈位低住佐佑佔何佗余佛作你佣佩佬佳併使侃來例侍供依侮侯侵侶侷便係促俊俎俏俐俗俘保俞俠信修俯俱俸倆倉個倌倍們倒倖倘候倚倜借倡倦倩倪倫值假偉偎偏偕做停健側偵偶偷偽傀傅傍傑傘備傢催傭傲傳債傷傻傾僅像僑僚僥僮僵價僻儀儂億儆儉儕儘償儡優儲儻儼允元兄充兆兇先光克兌免兒兔兜兢入內全兩八公六兮共兵其具典兼冀冊再冒冕冗冠冤冥冬冰冷准凌凍凜凝几凡凱凳凶凸凹出函刀刁分切刊刑划列初判別利刪刮到制刷券刺刻剁剃則剉削剋剌前剔剖剛剝剩剪剮副割創剷劃劇劈劉劍劑力功加劣助努劫勁勃勇勉勒動勘務勝勞募勢勤勳勵勸勻勾勿包匆化北匙匝匠匣匪匯匹匾匿區十千升午卉半卑卒卓協南博卜占卡卦卮卯印危即卵卷卸卻卿厄厚厝原厥厭厲去參又叉及友反叔取受叛叢口古句另叨叩只叫召叭叮可台史右司叼吃各合吉吊同名后吏吐向君吝吞吟否吧吩含吭吱吳吵吸吹吻吼吾呀呂呆呈告呎呢周呱味呵呸呼命咀咄和咎咐咒咕咖咚咧咩咪咫咬咱咳咻咽咾哀品哄哆哇哈哉哎哏員哥哦哨哩哪哭哲哺哼哽唆唇唉唐唧唬售唯唱唳唷唸唾啃商啊問啜啞啟啡啤啥啦啪啵啷啼啾喀喂善喇喉喊喔喘喚喜喝喪喬單喲喳喻嗅嗆嗇嗎嗑嗓嗚嗜嗝嗟嗡嗦嗨嗯嗲嗶嗽嘆嘉嘍嘔嘖嘗嘛嘟嘩嘮嘯嘰嘲嘴嘶嘻嘿噁噎噓噗噠噢器噩噪噬噱噴噸噹嚀嚇嚎嚏嚐嚓嚥嚨嚮嚴嚷嚼囂囉囊囑囚四囝回因囡囤困固圃圈國圍園圓圖團圜土在圭地圾址均坊坎坐坑坡坤坦坨坪坷垂垃型垮埃埋城埔域執培基堂堅堆堡堤堪報場堵塊塌塑塔塗塞填塵境墅墊墓墜增墨墮墳墾壁壇壓壕壘壞壤壩士壬壯壺壽夏夕外多夜夠夢夥大天太夫夭央夯失夷夾奇奈奉奏契奔奕套奠奢奧奪奮女奴奶奸她好如妃妄妒妖妙妝妞妥妨妮妳妹妻妾姆姊始姐姑姓委姚姜姦姨姪姬姻姿威娃娘娛娜娟娩娶娼婆婉婚婢婦婪婷婿媒媚媛媲媳媽嫁嫂嫉嫌嫖嫡嫣嫩嫵嬉嬋嬌嬤嬰嬸子孔孕字存孝孟季孤孩孫孬孰孳孵學孺孽它宅宇守安宋完宏宗官宙定宜客宣室宮宰害宴宵家容宿寂寄密富寐寒寓寞察寡寢實寧審寫寬寵寶寸寺封射將專尉尊尋對導小少尖尚尤尬就尷尺尼尾尿局屁居屆屈屋屌屍屎屏屑展屜屠屢層履屬山岩岱岳岸峰島崇崖崗崩崽嵌嶄嶺嶽巍川州巡巢工左巧巨巫差己已巴巷巾巿市布帆希帑帕帖帘帝帥師席帳帶帷常帽幄幅幌幕幗幟幣幫干平年幸幹幻幼幽幾庄庇床序底庖店府度座庫庭康庸廁廂廈廉廊廓廖廚廝廟廠廢廣廳延廷建弄弊式弒弓弔引弗弛弟弧弭弱張強彆彈彌彎彙形彤彥彩彪彬彭彰影彷役彼彿往征待徇很徊律後徐徑徒得徘從御徨復循徬微徵德徹徽心必忌忍志忘忙忠快忱念忽怎怒怕怖思怠怡急怦性怨怪恃恆恍恐恕恙恢恤恥恨恩恬恭息恰恿悄悅悉悍悔悚悟悠患您悱悲悴悶悸悼情惋惑惕惘惚惜惟惠惡惰惱想惶惹惺惻愁愈愉意愕愚愛感愣愧愫愷愾慈態慌慍慎慕慘慚慢慣慧慨慫慮慰慵慶慷慾憂憊憋憎憐憑憔憚憤憧憨憩憫憬憲憶憾懂懇懈應懊懦懲懵懶懷懸懺懼懾懿戀戈成我戒戕或戚戟截戮戰戲戳戴戶戾房所扁扇扈手才扎扒打扔托扛扣扭扮扯扳扶批扼找承技抄抉把抑抒抓投抖抗折抨披抬抱抵抹押抽拄拆拇拈拉拋拌拍拎拐拒拓拔拖拗拘拙拚招拜括拭拯拳拴拷拼拾拿持指按挑挖挨挪挫振挺挽挾捅捆捉捍捏捐捕捧捨捫捲捶捷捺捻掀掃授掉掌掏掐排掘掙掛掠採探接控推掩措掮掰揈揉揍描提插揚換握揣揩揪揭揮援揹搆損搏搐搓搔搖搗搜搞搥搧搪搬搭搶摔摘摟摧摩摯摳摸摺摻撂撇撈撐撒撓撕撞撤撥撩撫播撮撲撼撿擁擄擅擇擊擋操擎擒擔據擠擦擬擱擲擴擷擺擻擾攀攏攔攘攜攝攣攤攪攬支收攸改攻放政故效敏救敕敗敘教敞敢散敦敬敲整敵敷數斂斃文斌斑斗料斜斟斤斥斧斬斯新斷方於施旁旅旋族旗既日旦旨早旺旻昂昆昇昌明昏易昔星映春昧昨昭是時晃晉晒晚晝晨普景晰晴晶智晾暄暇暈暑暖暗暢暨暫暮暱暴暸曄曆曉曖曜曝曠曦曬曰曲曳更書曹曼曾替最會月有朋服朔朗望朝期木未末本朱朵朽杆李杏材村杖杜杞束杯杰東杵松板枉析枕林枚果枝枯架枷枸柄柏某染柔柙柢查柯柱柳柴柺柿栓栗校株核根格栽桃框案桌桑桶桿梁梅梓梗條梟梢梧梨梭梯械梳棄棉棋棍棒棕棗棘棚棟棧森棲棵棺椅植椎椒椪楊楓楚楞楣業極楷概榔榜榨榮榴構槌槍槐槓槤槽樁樂樑樓標樞樟模樣樸樹橇橋橘橙機橡橫檔檢檬檯檳檸櫃櫥櫻欄權欠次欣欲欺欽款歇歉歌歐歡止正此步武歧歪歲歷歸歹死殃殄殆殉殊殖殘殞殭殯殲段殷殺殼殿毀毅毆母每毒比毛毫毯氏民氓氛氣氧氯氰水永氾汀汁求汐汗汙汝汞江池污汪汰決汽沃沈沉沐沒沖沙沛沫沮河沸油治沼沾沿況泄泉泊泌法泛泡波泣泥注泯泰泱泳洋洗洛洞津洩洪洲活洽派流浩浪浮浴海浸涂消涉涕涮涯液涵涼淇淋淌淑淒淘淚淡淤淨淪淫淬深淵混淺添清渙減渝渠渡渣渦測港渲渴游渺渾湊湓湖湛湧湮湯溉源準溜溝溢溪溫溶溺溼滄滅滋滌滑滔滯滲滴滷滾滿漁漂漆漏漓演漠漢漣漩漪漫漬漱漲漸漿潑潔潘潛潢潤潭潮潯潰澀澄澆澈澎澡澤澱澳激濂濃濕濟濡濫濱濺濾瀆瀉瀏瀑瀚瀝瀟瀨瀾灌灑灘灣火灰灶灸灼災炊炎炒炙炫炭炮炯炳炷炸為烈烊烏烘烤烯烹焉焗焙焚無焢焦然煉煌煎煒煙煞煤煥照煩煮熄熊熙熟熬熱燃燈燉燒燕燙燜營燥燦燭燻爆爍爐爛爪爬爭爵父爸爹爺爽爾牆片版牌牙牛牟牠牡牢物牲特牽犀犒犧犬犯狀狂狐狒狗狙狠狡狩狸狹狼狽猈猙猛猜猥猩猴猶猾猿獄獅獎獒獨獰獲獵獷獸獻玄率玉王玩玫玲玷玻珈珊珍珠班珮現球理琪琮琳琴琵琶瑄瑋瑕瑙瑚瑜瑞瑤瑩瑪瑰瑾璀璃璇璞璣璧璨環瓊瓏瓜瓢瓣瓦瓶瓷甄甕甘甚甜生產用甩甭甯田由甲申男甸町界畏留畜畢略番畫異當畸疇疊疏疑疙疚疝疣疤疫疲疵疸疹疼疽疾痂病症痊痔痕痘痙痛痞痠痢痣痧痰痱痴痺痿瘀瘁瘋瘓瘟瘡瘤瘦瘩瘴療癌癒癖癟癡癢癥癩癮癱癲登發白百皂的皆皇皓皮皰皺盃盆盈益盒盔盛盜盞盟盡監盤盥盧盪目盯盲直相盼盾省眉看真眠眨眩眷眼眾睏睛睜睞睡督睦睪睫睹睽瞄瞇瞌瞎瞑瞞瞥瞧瞪瞬瞭瞻矇矚矛矜矣知矩矬短矮矯石矽砂砍研砣砦砰砲破砸硫硬硯碌碎碑碗碘碟碧碩碰碳碴確碼磁磊磋磕磚磨磷磺礁礎礙礦礪礬示社祂祈祉祐祕祖祚祝神祟祠祥票祭祺禁禍福禦禧禮禱禽禿秀私秉秋科秒秘租秤秦秩移稀稅程稍稚稜稟種稱稻稿穆積穎穗穠穢穩穫穴究空穿突窄窈窕窗窘窟窩窮窺竄竅竇竊立站竟章竣童竭端競竹竿笑笛符笨第筆等筊筋筍筐筒答策筠筱筵筷箇箋箍箏算管箭箱箴節範篇築篡篤篩篷簍簡簣簧簷簽簾簿籃籌籍籠籤籬籮籲米籽粉粒粗粥粹粽精糊糕糖糗糙糞糟糠糧糬糰系糾紀約紅紋納紓純紕紗紙級紛紜素索紫紮累細紳紹終組絆結絕絞絡給絨絮統絲綁經綜綠維綱網綴綵綸綺綻綽綿緊緋緒緘線緝締緣編緩緬緯練緻縛縝縣縫縮縱總績繁繃織繕繞繡繩繪繫繭繳繹繼續纏纖缸缺缽罄罈罐罔罕罩罪置罰署罵罷罹羅羈羊美羔羞群羥羨義羹羽翁翅翎習翔翠翡翩翹翻翼耀老考者而耍耐耑耕耗耙耳耶耽耿聆聊聖聘聚聞聯聰聲聳職聽聾肅肆肇肉肋肌肓肖肘肚肛肝股肢肥肩肪肯肱育肺胃背胎胖胚胛胞胡胰胱胳胸胺能脂脅脆脈脊脖脛脫脹脾腋腎腐腑腔腕腥腦腫腰腳腴腸腹腺腿膀膊膏膚膛膜膝膠膨膩膳膽膿臀臂臆臉臊臍臘臟臣臥臨自臬臭至致臺臼舀舅與興舉舊舌舍舒舖舞舟航般舶船艘艙艦良艱色艷艾芊芋芎芒芝芥芬芭芯花芳芸芹芽苑苗苛苞苟若苦苯英苺茁茂范茄茅茉茛茫茱茲茶茹草荊荒荷荼莉莊莎莒莓莖莞莧莫莽菁菇菊菌菜菠菩菪菫華菱菲菸萃萄萊萌萍萎萬萱落葉著葛葡董葩葫葬葳葵葷蒂蒐蒙蒜蒞蒲蒸蒼蓄蓉蓋蓓蓬蓮蓽蔓蔔蔗蔘蔡蔣蔥蔬蔭蔽蕁蕉蕨蕩蕭蕾薄薑薛薦薩薪薯薰藉藍藏藐藝藤藥藹蘆蘇蘊蘋蘭蘿虎虐處虛虜虞號虧虱虹蚊蚌蚓蚩蚪蚯蚱蚵蛀蛆蛇蛋蛙蛛蛟蛤蛻蜀蜂蜆蜊蜘蜚蜜蜢蝌蝕蝙蝟蝠蝦蝴蝶螂螃融螞螢螳螺蟆蟑蟬蟲蟹蟻蠅蠍蠟蠢蠶蠻血衊行衍術街衛衝衡衣表衫衰衷袁袋袍袒袖被袱裁裂裎裕裙補裝裡裳裴裸裹製複褥褫褲褻襖襠襪襬襯襲西要覆見規覓視覦親覬覺覽觀角解觸言訂計訊討訓訕託記訛訝訟訣訥訪設許訴診註詐評詛詞詠詢詣試詩詭詰話該詳詹誅誆誇誌認誓誕誘語誠誡誣誤誦誨說誰課誹誼調諂諄談諉請諒論諛諜諧諮諱諷諸諺諾謀謂謅謊謎謗謙講謝謠謫謬謹證譎識譚譜警譬譯議譴護譽讀變讓讚讞谷豁豆豈豐豔象豪豫豬豹豺貂貅貉貌貓貝貞負財貢貧貨販貪貫責貲貴貶買貸費貼貿賀賁賂賃賄資賈賊賒賓賜賞賠賢賣賤質賭賴賺購賽贅贈贊贏贓贖赤赦赫走赴起趁超越趕趙趟趣趨足趴趾跌跎跑距跟跡跤跨跩跪路跳跺踏踝踢踩踰踴踵踹蹂蹄蹈蹉蹊蹋蹚蹟蹤蹦蹧蹬蹭蹲蹶蹺躁躇躊躍躪身躬躲躺軀車軋軌軍軒軟軸軼軾較載輔輕輛輝輩輪輯輸輻輾輿轄轅轉轍轎轟辛辜辟辣辦辨辭辮辯辰辱農迂迅迎近返迢迦迪迫述迴迷迺追退送逃逅逆逍透逐逑途逕逗這通逛逝逞速造逢連逮週進逸逼逾遁遂遇遊運遍過遑道達違遙遛遜遞遠遣適遭遮遲遵遷選遺遼避邀邁邂還邊邏那邦邪邱邵郁郊郎郝郡部郭郵都鄉鄙鄧鄭鄰酌配酒酥酬酮酵酷酸醃醇醉醋醒醜醞醫醬釀釁采釋里重野量釐金釘針釣釦釵鈍鈔鈕鈞鈣鈴鉀鉅鉗鉛鉤銀銅銘銬銳銷鋌鋒鋪鋸鋼錄錘錚錠錢錦錮錯錶鍊鍋鍛鍥鍵鍾鎍鎖鎚鎮鏈鏗鏘鏟鏡鏢鏽鐘鐵鑄鑑鑣鑫鑰鑲鑼鑽鑿長門閃閉開閒間閘閡閨閱閻闆闊闌闔闕闖關闢阮阱防阻阿陀附陋陌降限院陣除陪陰陳陵陶陷陸陽隆隊隍階隔隕隘隙際障隨險隱隸隻雀雁雄雅集雇雌雎雕雖雙雜雞離難雨雪雯雲零雷電需霄霆震霉霍霓霖霜霞霧露霸霹霾靂靈青靖靚靜非靠面革靳靴靶鞋鞘鞠鞭韋韌韓韭音韻響頁頂項順須頌預頑頒頓頗領頡頤頭頰頸頹頻顆題額顎顏願顛類顧顫顯顱風颱颳颼飄飆飛食飢飪飭飯飲飼飽飾餃餅餉養餌餐餒餓餘餞餡館餵餾餿饅饋饌饒饕饞首香馨馬馭馮馴駁駐駑駕駙駛駝駟駭駿騎騖騙騰騷驃驅驕驗驚驟驢骨骯骰骺骼髁髒髓體髖高髦髮鬆鬍鬚鬢鬥鬧鬨鬩鬱鬼魁魂魄魅魏魔魚魠魯魷鮑鮪鮮鯊鯨鰍鰲鱉鱸鳥鳩鳳鳴鴉鴕鴛鴦鴨鴻鴿鵝鵡鵰鶩鶯鶴鷗鷸鷹鸚鸞鹹鹼鹽鹿麋麗麟麥麵麻麼黃黎黏黑默黛點黨黯鼓鼠鼻齊齒齡齣齪齷龍龐龜％）＋，ＸＹ',
        punctuations="\"%\'()*+-./?、「」％）＋， ",
        phonemes="",
        is_unique=True,
        is_sorted=True,
    ),
    phoneme_cache_path=None,
    precompute_num_workers=12,
    start_by_longest=True,
    datasets=DATASETS_CONFIG_LIST,
    cudnn_benchmark=False,
    max_audio_len=SAMPLE_RATE * MAX_AUDIO_LEN_IN_SECONDS,
    mixed_precision=False,
    test_sentences=[
        [
            "確定無疑是個沒經過任何規劃審批的違章建築",
            "mandarin_drama_4",
            None,
            "ma",
        ],
        [
            "貴賓酒樓",
            "mandarin_drama_200",
            None,
            "ma",
        ],
        [
            "目前市場上搭載英特爾芯片的智能手機不超過十款",
            "mandarin_drama_500",
            None,
            "ma",
        ],
        [
            "設計出來的家具耐受力也差",
            "mandarin_drama_354",
            None,
            "ma",
        ],
        [
            "動作迅速而引發廣泛關注.",
            "mandarin_drama_17",
            None,
            "ma",
        ],
    ],
    # Enable the weighted sampler
    use_weighted_sampler=False,
    # Ensures that all speakers are seen in the training batch equally no matter how many samples each speaker has
    weighted_sampler_attrs={"speaker_name": 1.0},
    weighted_sampler_multipliers={},
    # It defines the Speaker Consistency Loss (SCL) α to 9 like the paper
    speaker_encoder_loss_alpha=0.0,
)

# Load all the datasets samples and split traning and evaluation sets
train_samples, eval_samples = load_tts_samples(
    config.datasets,
    eval_split=True,
    eval_split_max_size=config.eval_split_max_size,
    eval_split_size=config.eval_split_size,
)

# Init the model
model = Vits.init_from_config(config)

# Init the trainer and 🚀
trainer = Trainer(
    TrainerArgs(restore_path=RESTORE_PATH, skip_train_epoch=SKIP_TRAIN_EPOCH),
    config,
    output_path=OUT_PATH,
    model=model,
    train_samples=train_samples,
    eval_samples=eval_samples,
    is_artifact=False,
)
trainer.fit()

#  python -m trainer.distribute --script train_yourtts.py --gpus "0,1,2,3"
# CUDA_VISIBLE_DEVICES=2 python TTS/bin/compute_embeddings.py --model_path tts_models--multilingual--multi-dataset--your_tts/model_se.pth.tar --config_path  tts_models--multilingual--multi-dataset--your_tts/config_se.json --config_dataset_path tts_models--multilingual--multi-dataset--your_tts/config.json --output_path tts_models--multilingual--multi-dataset--your_tts/vctk_d_vector_file.json
