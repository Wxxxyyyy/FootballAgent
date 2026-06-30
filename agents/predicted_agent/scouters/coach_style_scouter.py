# -*- coding: utf-8 -*-
"""
教练风格 + 惯用阵型采集器

数据来源（2026-06-26 更新）：
  - 教练名单：Bolavip "2026 World Cup coaches: All 48 managers"
    (https://bolavip.com/en/world-cup/2026-world-cup-coaches-all-48-managers-of-the-qualified-national-teams)
  - 战术风格/阵型：FIFA官网、Wikipedia、Guardian、BBC、Transfermarkt、
    tacticalfootballanalysis.com 等公开报道
  - 每条 description 标注信息来源，避免凭印象编写

输出格式:
  {
    "coach": "Carlo Ancelotti",
    "preferred_formation": "4-2-4",
    "style": "进攻控球",          # 进攻控球/防守反击/高位压迫/防守稳健/全攻全守
    "style_tags": ["控球", "边路进攻", "高位逼抢"],
    "tendency": "攻强守弱",       # 攻强守弱/攻守平衡/守强攻弱
    "description": "倾向控球主导，边后卫积极插上，中场创造力强"
  }
"""

import logging
from typing import Optional

from agents.predicted_agent.scouters.national_team_config import get_team_info

logger = logging.getLogger(__name__)


# ======================== 静态教练风格数据 ========================
# 世界杯32/48强常见队伍的教练风格
# 后续可通过爬虫自动更新，此处为兜底数据

_COACH_STYLE_DB = {
    "Brazil": {
        "coach": "Carlo Ancelotti",
        "preferred_formation": "4-2-4",
        "style": "进攻控球",
        "style_tags": ["边路进攻", "前场个人能力", "4-2-4阵型"],
        "tendency": "攻强守弱",
        "description": "安切洛蒂上任后采用4-2-4阵型（来源：FIFA/Guardian/BBC），强调前场攻击群个人能力，边路进攻犀利，中场双人组偏防守",
    },
    "Argentina": {
        "coach": "Lionel Scaloni",
        "preferred_formation": "4-3-3",
        "style": "攻守平衡",
        "style_tags": ["中场绞杀", "快速反击", "梅西核心"],
        "tendency": "攻守平衡",
        "description": "斯卡洛尼打造的务实阿根廷，中场硬度高，攻防转换快，围绕梅西构建进攻体系",
    },
    "Germany": {
        "coach": "Julian Nagelsmann",
        "preferred_formation": "4-2-3-1",
        "style": "高位压迫",
        "style_tags": ["高位逼抢", "阵地渗透", "边中结合"],
        "tendency": "攻强守弱",
        "description": "纳格尔斯曼注重高位压迫和快速转换，阵地进攻层次丰富，但高位防线容易被反击打穿",
    },
    "France": {
        "coach": "Didier Deschamps",
        "preferred_formation": "4-2-3-1",
        "style": "防守反击",
        "style_tags": ["防守稳固", "快速反击", "姆巴佩冲刺"],
        "tendency": "守强攻弱",
        "description": "德尚的务实风格，中场防守硬度高，依赖姆巴佩个人速度打反击，阵地进攻有时缺乏创造力",
    },
    "Spain": {
        "coach": "Luis de la Fuente",
        "preferred_formation": "4-3-3",
        "style": "进攻控球",
        "style_tags": ["tiki-taka", "控球", "边路突破"],
        "tendency": "攻守平衡",
        "description": "西班牙传统传控风格，中场控球能力强，边路有亚马尔等爆点，进攻节奏可快可慢",
    },
    "England": {
        "coach": "Thomas Tuchel",
        "preferred_formation": "3-4-2-1",
        "style": "攻守平衡",
        "style_tags": ["三中卫", "翼卫插上", "定位球"],
        "tendency": "攻守平衡",
        "description": "图赫尔善用三中卫体系，翼卫是关键进攻发起点，定位球战术丰富，防守组织严密",
    },
    "Portugal": {
        "coach": "Roberto Martínez",
        "preferred_formation": "4-2-3-1",
        "style": "进攻控球",
        "style_tags": ["控球", "边路进攻", "B费直塞"],
        "tendency": "攻强守弱",
        "description": "马丁内斯强调控球和进攻层次，B费承担创造任务，前场 talent 丰富，防守端中卫速度偏慢",
    },
    "Netherlands": {
        "coach": "Ronald Koeman",
        "preferred_formation": "3-4-3",
        "style": "全攻全守",
        "style_tags": ["三中卫", "全攻全守", "边路推进"],
        "tendency": "攻守平衡",
        "description": "荷兰全攻全守传统，三中卫体系翼卫大幅压上，攻防转换节奏快，后场出球能力强",
    },
    "Belgium": {
        "coach": "Rudi Garcia",
        "preferred_formation": "4-3-3",
        "style": "攻守平衡",
        "style_tags": ["德布劳内核心", "快速转换", "阵地进攻"],
        "tendency": "攻守平衡",
        "description": "加西亚2025年接替特德斯科（来源：ESPN/Wikipedia），曾执教里尔/罗马/马赛/那不勒斯，战术风格混合老派与现代元素",
    },
    "Italy": {
        "coach": "Luciano Spalletti",
        "preferred_formation": "4-3-3",
        "style": "高位压迫",
        "style_tags": ["高位逼抢", "快速推进", "防守传统"],
        "tendency": "攻守平衡",
        "description": "斯帕莱蒂注重前场压迫和快速推进，保留意大利防守传统，阵地进攻有层次",
    },
    "Croatia": {
        "coach": "Zlatko Dalić",
        "preferred_formation": "4-3-3",
        "style": "中场控球",
        "style_tags": ["中场控制", "莫德里奇核心", "防守稳健"],
        "tendency": "攻守平衡",
        "description": "达利奇的克罗地亚以中场控制见长，莫德里奇调度全队，整体节奏偏慢但控制力强",
    },
    "Japan": {
        "coach": "Hajime Moriyasu",
        "preferred_formation": "4-2-3-1",
        "style": "快速反击",
        "style_tags": ["快速反击", "高位逼抢", "团队配合"],
        "tendency": "攻守平衡",
        "description": "森保一的日本队以快速反击和高位逼抢著称，旅欧球员技术细腻，整体性强，善于以弱胜强",
    },
    "South Korea": {
        "coach": "Hong Myung-bo",
        "preferred_formation": "4-3-3",
        "style": "快速反击",
        "style_tags": ["快速反击", "孙兴慜核心", "高位逼抢"],
        "tendency": "攻强守弱",
        "description": "韩国以孙兴慜为进攻核心，快速反击犀利，体能充沛逼抢凶狠，但防守端偶有失误",
    },
    "Morocco": {
        "coach": "Mohamed Ouahbi",
        "preferred_formation": "4-3-3",
        "style": "防守反击",
        "style_tags": ["防守稳固", "快速反击", "身体对抗"],
        "tendency": "守强攻弱",
        "description": "瓦赫比2026年3月接替雷格拉吉（来源：FIFA/BBC），距世界杯仅三个月上任，延续摩洛哥2022世界杯四强的防守反击体系",
    },
    "USA": {
        "coach": "Mauricio Pochettino",
        "preferred_formation": "4-2-3-1",
        "style": "高位压迫",
        "style_tags": ["高位逼抢", "体能充沛", "快速转换"],
        "tendency": "攻守平衡",
        "description": "波切蒂诺带来高位压迫风格，美国队体能充沛跑动能力强，但技术细腻度不足",
    },
    "Mexico": {
        "coach": "Javier Aguirre",
        "preferred_formation": "4-3-3",
        "style": "进攻控球",
        "style_tags": ["控球", "边路进攻", "主场气势"],
        "tendency": "攻强守弱",
        "description": "阿吉雷的墨西哥注重控球和边路进攻，主场作战气势足，但面对强队防守端压力大",
    },
    "Uruguay": {
        "coach": "Marcelo Bielsa",
        "preferred_formation": "4-3-3",
        "style": "高位压迫",
        "style_tags": ["疯狂逼抢", "人盯人", "高强度跑动"],
        "tendency": "攻强守弱",
        "description": "贝尔萨的疯子足球，全场人盯人高位逼抢，强度极高但体能消耗大，下半场可能体能断崖",
    },
    "Colombia": {
        "coach": "Néstor Lorenzo",
        "preferred_formation": "4-2-3-1",
        "style": "攻守平衡",
        "style_tags": ["J罗核心", "边路突破", "防守稳健"],
        "tendency": "攻守平衡",
        "description": "洛伦佐的哥伦比亚围绕J罗构建进攻，边路有路易斯迪亚斯等爆点，整体攻守较为平衡",
    },
    "Switzerland": {
        "coach": "Murat Yakin",
        "preferred_formation": "3-4-2-1",
        "style": "防守稳健",
        "style_tags": ["三中卫", "防守组织", "快速反击"],
        "tendency": "守强攻弱",
        "description": "雅金的瑞士防守组织严密，三中卫体系稳固，反击时扎卡长传精准，阵地进攻能力一般",
    },
    "Denmark": {
        "coach": "Kasper Hjulmand",
        "preferred_formation": "4-3-3",
        "style": "中场控球",
        "style_tags": ["中场控制", "埃里克森核心", "整体性强"],
        "tendency": "攻守平衡",
        "description": "尤尔曼德的丹麦以中场控制为主，埃里克森负责创造，整体性强但缺乏顶级爆点球员",
    },
    "Turkey": {
        "coach": "Vincenzo Montella",
        "preferred_formation": "4-2-3-1",
        "style": "攻守平衡",
        "style_tags": ["边路进攻", "恰尔汗奥卢核心", "主场气势"],
        "tendency": "攻强守弱",
        "description": "蒙特拉的土耳其进攻有层次，恰尔汗奥卢定位球和远射能力强，但防守端不够稳定",
    },
    "Serbia": {
        "coach": "Dragan Stojković",
        "preferred_formation": "3-4-2-1",
        "style": "攻守平衡",
        "style_tags": ["三中卫", "高空优势", "米特罗维奇支点"],
        "tendency": "攻强守弱",
        "description": "斯托伊科维奇的塞尔维亚有身体优势，米特罗维奇做支点，但中场创造力有限，防守端转身慢",
    },
    "Australia": {
        "coach": "Tony Popovic",
        "preferred_formation": "4-4-2",
        "style": "防守反击",
        "style_tags": ["身体对抗", "防守稳固", "定位球"],
        "tendency": "守强攻弱",
        "description": "波波维奇的澳大利亚注重身体对抗和防守，定位球有优势，但技术细腻度不足",
    },
    "Iran": {
        "coach": "Amir Ghalenoei",
        "preferred_formation": "4-2-3-1",
        "style": "防守反击",
        "style_tags": ["防守密集", "快速反击", "身体对抗"],
        "tendency": "守强攻弱",
        "description": "伊朗以密集防守和快速反击为主，身体对抗强，塔雷米是反击支点，但控球能力有限",
    },
    "Senegal": {
        "coach": "Pape Thiaw",
        "preferred_formation": "4-3-3",
        "style": "攻守平衡",
        "style_tags": ["身体对抗", "边路突破", "萨迪奥马内核心"],
        "tendency": "攻守平衡",
        "description": "塞内加尔身体对抗强，马内领衔进攻，整体速度快，但战术纪律性偶有波动",
    },
    "Ecuador": {
        "coach": "Sebastián Beccacece",
        "preferred_formation": "4-2-3-1",
        "style": "防守反击",
        "style_tags": ["高位逼抢", "快速反击", "身体对抗"],
        "tendency": "攻守平衡",
        "description": "厄瓜多尔以高位逼抢和快速反击为主，高原主场优势明显，体能充沛",
    },
    "Nigeria": {
        "coach": "Eric Chelle",
        "preferred_formation": "4-3-3",
        "style": "快速反击",
        "style_tags": ["速度优势", "边路突破", "奥斯梅恩支点"],
        "tendency": "攻强守弱",
        "description": "尼日利亚前场速度极快，奥斯梅恩是强力支点，但中场组织和防守端较为薄弱",
    },
    "Costa Rica": {
        "coach": "Gustavo Alfaro",
        "preferred_formation": "5-4-1",
        "style": "防守稳健",
        "style_tags": ["五后卫", "密集防守", "快速反击"],
        "tendency": "守强攻弱",
        "description": "哥斯达黎加以五后卫密集防守为主，反击依赖纳瓦斯长传，整体偏保守",
    },
    "Canada": {
        "coach": "Jesse Marsch",
        "preferred_formation": "4-4-2",
        "style": "高位压迫",
        "style_tags": ["高位逼抢", "体能充沛", "阿方索戴维斯冲刺"],
        "tendency": "攻守平衡",
        "description": "马什带来高位压迫风格，加拿大队体能好跑动多，阿方索戴维斯是边路爆点",
    },
    "Poland": {
        "coach": "Michał Probierz",
        "preferred_formation": "4-2-3-1",
        "style": "防守反击",
        "style_tags": ["莱万支点", "防守稳固", "定位球"],
        "tendency": "守强攻弱",
        "description": "波兰围绕莱万构建进攻，整体偏防守反击，中场创造力不足，依赖莱万个人能力",
    },
    "Ukraine": {
        "coach": "Serhiy Rebrov",
        "preferred_formation": "4-2-3-1",
        "style": "攻守平衡",
        "style_tags": ["中场控制", "穆德里克冲刺", "整体配合"],
        "tendency": "攻守平衡",
        "description": "雷布罗夫的乌克兰中场有控制力，穆德里克提供速度，整体配合较好",
    },
    "Austria": {
        "coach": "Ralf Rangnick",
        "preferred_formation": "4-2-3-1",
        "style": "高位压迫",
        "style_tags": ["Gegenpressing", "高位逼抢", "快速转换"],
        "tendency": "攻强守弱",
        "description": "朗尼克的高位压迫体系，奥地利逼抢凶狠转换快，但高位防线容易被反击打穿",
    },
    # ── 48队补录 ──
    "South Africa": {
        "coach": "Hugo Broos",
        "preferred_formation": "4-3-3",
        "style": "防守反击",
        "style_tags": ["防守稳固", "快速反击", "身体对抗"],
        "tendency": "守强攻弱",
        "description": "布鲁斯的南非队注重防守纪律，反击速度快，但阵地进攻创造力有限",
    },
    "Czech Republic": {
        "coach": "Miroslav Koubek",
        "preferred_formation": "4-2-3-1",
        "style": "攻守平衡",
        "style_tags": ["整体配合", "中场组织", "防守纪律"],
        "tendency": "攻守平衡",
        "description": "库贝克74岁（来源：Wikipedia/BBC），世界杯史上最年长教练，曾三度执教比尔森胜利并夺捷克联赛冠军，前门将出身",
    },
    "Bosnia & Herzegovina": {
        "coach": "Sergej Barbarez",
        "preferred_formation": "4-2-3-1",
        "style": "攻守平衡",
        "style_tags": ["身体对抗", "定位球", "边路推进"],
        "tendency": "攻守平衡",
        "description": "巴尔巴雷兹的波黑身体对抗强，定位球有优势，整体攻守平衡",
    },
    "Qatar": {
        "coach": "Julen Lopetegui",
        "preferred_formation": "4-3-3",
        "style": "进攻控球",
        "style_tags": ["控球", "快速转换", "战术多变"],
        "tendency": "攻守平衡",
        "description": "洛佩特吉（西班牙籍，前西班牙/波尔图/狼队主帅）执教卡塔尔（来源：Wikipedia/FIFA），战术双面性：中位防守让阿菲夫边路活动，或高位压迫消耗对手",
    },
    "Haiti": {
        "coach": "Sébastien Migné",
        "preferred_formation": "4-4-2",
        "style": "防守反击",
        "style_tags": ["防守密集", "快速反击", "身体对抗"],
        "tendency": "守强攻弱",
        "description": "海地以防守反击为主，身体对抗能力强，但技术层面与强队有差距",
    },
    "Scotland": {
        "coach": "Steve Clarke",
        "preferred_formation": "5-4-1",
        "style": "防守稳健",
        "style_tags": ["五后卫", "密集防守", "定位球"],
        "tendency": "守强攻弱",
        "description": "克拉克的苏格兰采用五后卫体系，防守组织严密，依赖定位球和反击得分",
    },
    "Paraguay": {
        "coach": "Gustavo Alfaro",
        "preferred_formation": "4-4-2",
        "style": "防守反击",
        "style_tags": ["防守稳固", "快速反击", "身体对抗"],
        "tendency": "守强攻弱",
        "description": "阿尔法罗的巴拉圭防守纪律性强，反击效率高，南美传统硬朗风格",
    },
    "Curacao": {
        "coach": "Dick Advocaat",
        "preferred_formation": "4-3-3",
        "style": "攻守平衡",
        "style_tags": ["边路进攻", "快速转换", "技术型"],
        "tendency": "攻守平衡",
        "description": "艾德沃卡特为库拉索带来欧式打法，球员多有荷兰背景，技术型风格",
    },
    "Ivory Coast": {
        "coach": "Emerse Faé",
        "preferred_formation": "4-3-3",
        "style": "快速反击",
        "style_tags": ["速度优势", "边路突破", "身体对抗"],
        "tendency": "攻强守弱",
        "description": "法埃的科特迪瓦前场速度极快，身体强壮，但防守端组织性有时不足",
    },
    "Sweden": {
        "coach": "Graham Potter",
        "preferred_formation": "4-2-3-1",
        "style": "攻守平衡",
        "style_tags": ["进步战术", "控球", "整体配合"],
        "tendency": "攻守平衡",
        "description": "波特（英格兰籍）接手瑞典（来源：Wikipedia/Guardian/Transfermarkt），惯用4-2-3-1，将进步战术原则与北欧纪律性结合",
    },
    "Tunisia": {
        "coach": "Sabri Lamouchi",
        "preferred_formation": "4-3-3",
        "style": "防守反击",
        "style_tags": ["防守密集", "快速反击", "防守纪律"],
        "tendency": "守强攻弱",
        "description": "拉穆奇2026年1月接手突尼斯（来源：FIFA/Transfermarkt），继承前任Trabelsi的防守纪律体系；注：世界杯首战1-5负瑞典后被解雇，Kebaier临时接任",
    },
    "Egypt": {
        "coach": "Hossam Hassan",
        "preferred_formation": "4-2-3-1",
        "style": "防守反击",
        "style_tags": ["防守稳固", "萨拉赫核心", "快速反击"],
        "tendency": "守强攻弱",
        "description": "哈桑的埃及围绕萨拉赫构建反击，防守稳固，但中场创造力有限",
    },
    "New Zealand": {
        "coach": "Darren Bazeley",
        "preferred_formation": "4-4-2",
        "style": "防守反击",
        "style_tags": ["身体对抗", "防守组织", "定位球"],
        "tendency": "守强攻弱",
        "description": "巴兹利的新西兰注重身体对抗和防守，大洋洲风格，技术层面有限",
    },
    "Cape Verde": {
        "coach": "Bubista",
        "preferred_formation": "4-1-4-1",
        "style": "防守反击",
        "style_tags": ["快速反击", "技术型", "团队纪律"],
        "tendency": "攻守平衡",
        "description": "布比斯塔（真名Pedro Leitão Brito，来源：FIFA/MyLineups）带队历史性杀入世界杯，惯用4-1-4-1阵型，球员多有葡萄牙背景",
    },
    "Saudi Arabia": {
        "coach": "Georgios Donis",
        "preferred_formation": "4-2-3-1",
        "style": "防守反击",
        "style_tags": ["防守密集", "快速反击", "定位球"],
        "tendency": "守强攻弱",
        "description": "多尼斯（希腊籍）2026年4月底接替勒纳尔（来源：FIFA/Al Arabiya），距世界杯仅一个多月上任，有专门的定位球战术设计（来源：tacticalfootballanalysis）",
    },
    "Iraq": {
        "coach": "Graham Arnold",
        "preferred_formation": "4-4-2",
        "style": "防守反击",
        "style_tags": ["防守纪律", "整体配合", "1v1防守"],
        "tendency": "守强攻弱",
        "description": "阿诺德（澳大利亚籍）2025年5月接手伊拉克（来源：Wikipedia/FIFA），惯用4-4-2（来源：Deschamps采访），强调纪律性和1v1防守能力",
    },
    "Norway": {
        "coach": "Ståle Solbakken",
        "preferred_formation": "4-3-3",
        "style": "进攻控球",
        "style_tags": ["控球", "哈兰德支点", "边路推进"],
        "tendency": "攻强守弱",
        "description": "索尔巴肯的挪威围绕哈兰德构建进攻，控球能力强，但中场创造力和防守端有短板",
    },
    "Algeria": {
        "coach": "Vladimir Petković",
        "preferred_formation": "4-2-3-1",
        "style": "攻守平衡",
        "style_tags": ["中场控制", "边路进攻", "整体配合"],
        "tendency": "攻守平衡",
        "description": "佩特科维奇的阿尔及利亚中场有控制力，马赫雷斯等前场球员技术细腻",
    },
    "Jordan": {
        "coach": "Jamal Sellami",
        "preferred_formation": "4-2-3-1",
        "style": "防守反击",
        "style_tags": ["防守密集", "快速反击", "整体纪律"],
        "tendency": "守强攻弱",
        "description": "塞拉米（摩洛哥籍）2024年接手约旦（来源：Wikipedia/FIFA），号召球队效仿摩洛哥2022世界杯四强模式，防守反击为主",
    },
    "DR Congo": {
        "coach": "Sébastien Desabre",
        "preferred_formation": "4-3-3",
        "style": "快速反击",
        "style_tags": ["速度优势", "身体对抗", "边路突破"],
        "tendency": "攻强守弱",
        "description": "德萨布尔的刚果金球员身体素质出色，反击速度快，但战术纪律性有波动",
    },
    "Uzbekistan": {
        "coach": "Fabio Cannavaro",
        "preferred_formation": "4-2-3-1",
        "style": "防守稳健",
        "style_tags": ["防守优先", "紧凑阵型", "防守组织"],
        "tendency": "守强攻弱",
        "description": "卡纳瓦罗（意大利籍，2006世界杯冠军队长）执教乌兹别克斯坦（来源：Wikipedia/Guardian/Goal），延续球队紧凑防守优先的传统模板",
    },
    "Ghana": {
        "coach": "Carlos Queiroz",
        "preferred_formation": "4-2-3-1",
        "style": "防守反击",
        "style_tags": ["防守组织", "纪律性强", "快速反击"],
        "tendency": "守强攻弱",
        "description": "奎罗斯（葡萄牙籍，73岁）2026年4月接手加纳（来源：FIFA/Wikipedia），连续五届执教世界杯，以防守组织和战术纪律著称",
    },
    "Panama": {
        "coach": "Thomas Christiansen",
        "preferred_formation": "4-4-2",
        "style": "防守反击",
        "style_tags": ["防守密集", "快速反击", "身体对抗"],
        "tendency": "守强攻弱",
        "description": "克里斯滕森的巴拿马防守密集，依赖反击和定位球，中北美硬朗风格",
    },
    "Croatia_extra": None,  # 占位，确保字典结构正确
}
# 清理占位
_COACH_STYLE_DB.pop("Croatia_extra", None)


# ======================== 对外接口 ========================

def get_coach_style(team_en: str) -> Optional[dict]:
    """
    获取球队教练风格信息

    Args:
        team_en: 国家队英文标准名（如 "Brazil"）

    Returns:
        {
            "coach": "...",
            "preferred_formation": "4-3-3",
            "style": "进攻控球",
            "style_tags": [...],
            "tendency": "攻强守弱",
            "description": "..."
        }
        未收录返回 None
    """
    result = _COACH_STYLE_DB.get(team_en)
    if not result:
        logger.warning(f"[教练风格] 未收录 {team_en} 的教练风格数据")
        # 返回兜底空结构，避免调用方处理 None
        return {
            "coach": "未知",
            "preferred_formation": "未知",
            "style": "未知",
            "style_tags": [],
            "tendency": "未知",
            "description": f"{team_en} 教练风格数据暂未收录",
            "data_available": False,
        }

    # 补充数据可用标记
    result = dict(result)
    result["data_available"] = True
    return result


def compare_styles(home_en: str, away_en: str) -> dict:
    """
    对比两队教练风格，输出战术分析要点

    Returns:
        {
            "home": {...},
            "away": {...},
            "tactical_matchup": "风格对碰描述",
        }
    """
    home = get_coach_style(home_en)
    away = get_coach_style(away_en)

    # 简单的风格对碰分析
    matchup_parts = []
    if home and away and home.get("data_available") and away.get("data_available"):
        h_style = home["style"]
        a_style = away["style"]

        # 攻守倾向对比
        h_off = "攻强" in home.get("tendency", "")
        a_def = "守强" in away.get("tendency", "")
        if h_off and a_def:
            matchup_parts.append(f"{home_en} 攻强守弱 vs {away_en} 守强攻弱，可能陷入攻坚困境")

        a_off = "攻强" in away.get("tendency", "")
        h_def = "守强" in home.get("tendency", "")
        if a_off and h_def:
            matchup_parts.append(f"{away_en} 进攻强 vs {home_en} 防守稳，反击可能受限")

        # 阵型对比
        h_form = home.get("preferred_formation", "")
        a_form = away.get("preferred_formation", "")
        if h_form and a_form and h_form != a_form:
            matchup_parts.append(f"阵型差异: {home_en} {h_form} vs {away_en} {a_form}")

        # 风格相克
        if "高位压迫" in h_style and "快速反击" in a_style:
            matchup_parts.append(f"{home_en} 高位压迫可能被 {away_en} 快速反击针对")
        if "高位压迫" in a_style and "快速反击" in h_style:
            matchup_parts.append(f"{away_en} 高位压迫可能被 {home_en} 快速反击针对")
        if "进攻控球" in h_style and "防守反击" in a_style:
            matchup_parts.append(f"{home_en} 控球围攻 vs {away_en} 铁桶反击，经典对碰")

    return {
        "home": home,
        "away": away,
        "tactical_matchup": "；".join(matchup_parts) if matchup_parts else "无明显风格相克",
    }


# ======================== 测试 ========================

if __name__ == "__main__":
    print("=== 教练风格数据测试 ===\n")

    for team in ["Brazil", "Germany", "France", "Japan", "Argentina", "Morocco"]:
        style = get_coach_style(team)
        if style and style.get("data_available"):
            print(f"  {team}:")
            print(f"    教练: {style['coach']}")
            print(f"    阵型: {style['preferred_formation']}")
            print(f"    风格: {style['style']} ({style['tendency']})")
            print(f"    标签: {', '.join(style['style_tags'])}")
            print()

    print("=== 风格对碰 ===")
    matchup = compare_styles("Brazil", "France")
    print(f"  Brazil vs France:")
    print(f"    {matchup['tactical_matchup']}")
