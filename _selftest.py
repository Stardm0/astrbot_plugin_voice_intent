# -*- coding: utf-8 -*-
"""开口吧 · 离线自检（纯标准库，不连 AstrBot、不调模型、不花 token）

    python _selftest.py
"""
from __future__ import annotations

import asyncio
import base64
import importlib.util
import json
import logging
import os
import sys
import tempfile
import types

HERE = os.path.dirname(os.path.abspath(__file__))
MAIN = os.path.join(HERE, "main.py")

# ------------------------------------------------------------------ 替身

class Plain:
    def __init__(self, text=""):
        self.text = text

    def __repr__(self):
        return f"Plain({self.text!r})"


class Record:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __repr__(self):
        return f"Record({self.kwargs.keys()})"


class Image:
    pass


class _FakeResult:
    def __init__(self, chain):
        self.chain = chain


class FakeEvent:
    def __init__(self, text="", umo="aiocqhttp:FriendMessage:10001", chain=None):
        self.message_str = text
        self.unified_msg_origin = umo
        self._result = _FakeResult(list(chain) if chain is not None else [])
        self.sent = []

    def get_result(self):
        return self._result

    def plain_result(self, text):
        self.sent.append(text)
        return text

    def chain_result(self, chain):
        self.sent.append(("chain", list(chain)))
        return ("chain", list(chain))


class FakeProvider:
    def __init__(self, payload=None, boom=False):
        self.payload = payload
        self.boom = boom
        self.calls = []

    async def get_audio(self, text):
        self.calls.append(text)
        if self.boom:
            raise RuntimeError("provider down")
        return self.payload


class FakeTTS:
    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    async def tts(self, text):
        self.calls.append(text)
        return self.payload

    async def __call__(self, text):
        return await self.tts(text)


class FakeContext:
    def __init__(self, tts=None, provider=None):
        if tts is not None:
            self.tts = tts
        self.provider_manager = types.SimpleNamespace(curr_tts_provider_inst=provider)

    async def get_current_chat_provider_id(self, umo=""):
        return "fake-provider"

    async def llm_generate(self, **kwargs):
        return types.SimpleNamespace(completion_text='{"want": false, "scope": "once", "off": false}')


def _stub(name, **attrs):
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    sys.modules[name] = mod
    return mod


class _Deco:
    def on_decorating_result(self, *a, **k):
        def deco(fn):
            return fn
        return deco

    def command(self, *a, **k):
        def deco(fn):
            return fn
        return deco

    def __getattr__(self, _):
        def deco(*a, **k):
            def wrap(fn):
                return fn
            return wrap
        return deco


logging.basicConfig(level=logging.CRITICAL)
_stub("astrbot", api=types.ModuleType("astrbot.api"))
_stub("astrbot.api", logger=logging.getLogger("voice_test"))
_stub("astrbot.api.event", AstrMessageEvent=FakeEvent, filter=_Deco())
class FakeStar:
    def __init__(self, context=None, *args, **kwargs):
        self.context = context


_stub(
    "astrbot.api.star",
    Context=object,
    Star=FakeStar,
    register=lambda *a, **k: (lambda cls: cls),
    StarTools=types.SimpleNamespace(get_data_dir=lambda *a, **k: tempfile.mkdtemp(prefix="voice_test_")),
)
_stub("astrbot.api.message_components", Plain=Plain, Record=Record, Image=Image)

spec = importlib.util.spec_from_file_location("voice_intent_mod", MAIN)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

# ------------------------------------------------------------------ 断言框架

PASSED, FAILED = [], []


def check(name, cond, detail=""):
    if cond:
        PASSED.append(name)
    else:
        FAILED.append((name, detail))
        print("FAIL " + name + ("  " + str(detail) if detail else ""))


def new_plugin(**cfg):
    ctx = FakeContext(
        tts=FakeTTS(base64.b64encode(b"FAKEAUDIO").decode()),
        provider=FakeProvider(payload=base64.b64encode(b"PROVIDERAUDIO-PAYLOAD-1234567890").decode()),
    )
    p = mod.VoiceIntentPlugin(ctx, cfg)
    return p, ctx


def run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


U = "aiocqhttp:FriendMessage:10001"


# ------------------------------------------------------------------ 1 元数据 / 配置

print("== 1 元数据与配置 ==")
meta = {}
for line in open(os.path.join(HERE, "metadata.yaml"), encoding="utf-8"):
    if ":" in line and not line.strip().startswith("#"):
        k, v = line.split(":", 1)
        meta[k.strip()] = v.strip().strip('"')
check("metadata name", meta.get("name") == "astrbot_plugin_voice_intent", meta)
check("metadata 有 desc", len(meta.get("desc", "")) > 60, meta.get("desc"))
check("metadata 有版本", bool(meta.get("version")), meta)
check("版本号保持 1.0.0", meta.get("version") == "1.0.0", meta.get("version"))
main_src = open(MAIN, encoding="utf-8").read()
check("register 版本仍是 1.0.0", '"1.0.0"' in main_src)
check("全仓无 1.1.0 残留", "1.1.0" not in main_src and "1.1.0" not in open(os.path.join(HERE, "metadata.yaml"), encoding="utf-8").read())

schema = json.load(open(os.path.join(HERE, "_conf_schema.json"), encoding="utf-8"))
for key in ("enabled", "mode", "hints", "off_hints", "judge_provider_id", "max_tts_chars",
            "strip_brackets", "attach_text", "session_timeout_minutes"):
    check("schema 有 " + key, key in schema)
check("mode 默认 keyword", schema["mode"]["default"] == "keyword", schema["mode"]["default"])
check("attach_text 默认关", schema["attach_text"]["default"] is False)

# ------------------------------------------------------------------ 2 指令解析

print("== 2 指令解析 ==")
p, _ = new_plugin()
cases = [
    ("/voice", ("", "")),
    ("/voice 状态", ("状态", "")),
    ("/voice 判定 llm", ("判定", "llm")),
    ("/开口 帮助", ("帮助", "")),
    ("/开口 开", ("开", "")),
    ("状态", ("状态", "")),
    ("判定 always", ("判定", "always")),
    ("判定  llm  ", ("判定", "llm")),
    ("／voice 关", ("关", "")),
    ("\ufeff/voice\u200b 一次", ("一次", "")),
    ("[CQ:at,qq=1] /voice 清", ("清", "")),
    ("/开口吧 自检", ("自检", "")),
]
for whole, want in cases:
    check("解析 " + repr(whole), p._parse_cmd(whole) == want, (whole, p._parse_cmd(whole), want))

# ------------------------------------------------------------------ 3 关键词与关闭

print("== 3 关键词与关闭 ==")
p, _ = new_plugin()
for text in ("用语音跟我说晚安", "发条语音吧", "念给我听", "别打字了，直接说",
             "不想看字了", "你开口说啊", "SPEAK to me", "来段语音"):
    check("命中关键词 " + text, p._keyword_hit(text), text)
for text in ("今天天气真闷啊", "给我讲个故事", "说句晚安",
             "帮我看看这段代码"):
    check("不误命中 " + text, not p._keyword_hit(text), text)
for text in ("别发语音了", "还是打字吧", "不用语音", "关掉语音", "NO VOICE"):
    check("识别关闭 " + text, p._is_off(text), text)
check("关闭优先于命中", run(p._decide(FakeEvent("别发语音了", U), "别发语音了", U))["voice"] is False
      and run(p._decide(FakeEvent("别发语音了", U), "别发语音了", U))["off"] is True)
check("custom extra_regex", new_plugin(extra_regex=r"用嗓子")[0]._keyword_hit("用嗓子讲"))
check("非法正则不炸", new_plugin(extra_regex="(")[0]._keyword_hit("用语音说"))

# ------------------------------------------------------------------ 4 闸门三档

print("== 4 闸门三档 ==")
pkw, _ = new_plugin(mode="keyword")
pllm, _ = new_plugin(mode="llm")
palw, _ = new_plugin(mode="always")
check("keyword: 闲聊零调用", not pkw._gate("今天天气真闷啊", U))
check("keyword: 明确要语音才召回", pkw._gate("用语音跟我说", U))
check("keyword: 短字放行", not pkw._gate("嗯", U))
check("keyword: 斜杠放行", not pkw._gate("/help", U))
check("keyword: 纯表情放行", not pkw._gate("[旺柴]", U))
check("llm: 闲聊零调用", not pllm._gate("今天天气真闷啊", U))
check("llm: 隐含线索召回", pllm._gate("懒得看，你直接讲吧", U))
check("llm: 明确说法召回", pllm._gate("念给我听", U))
check("always: 每句都判", palw._gate("今天天气真闷啊", U))
check("llm: 关闭词召回", pllm._gate("还是打字吧", U))

# ------------------------------------------------------------------ 5 文本清洗

print("== 5 文本清洗 ==")
p, _ = new_plugin()
check("删表情标记", p._clean_for_tts("好呀[旺柴]我这就去") == "好呀我这就去",
      p._clean_for_tts("好呀[旺柴]我这就去"))
check("保留非表情方括号", "PROTOCOL_ENFORCEMENT" in p._clean_for_tts("看 `PROTOCOL_ENFORCEMENT` 这个标记"))
check("删静默暗号", p._clean_for_tts("[NO_REPLY]") == "")
check("删 think", p._clean_for_tts("<think>嗯…</think>晚安") == "晚安",
      p._clean_for_tts("<think>嗯…</think>晚安"))
check("删零宽", p._clean_for_tts("晚\u200b安") == "晚安")
check("多空白归一", p._clean_for_tts("好   呀") == "好 呀")
chain = [Plain("第一句。"), Image(), Plain("第二句[旺柴]。")]
text, others = p._collect_text(chain)
check("多 Plain 拼接", text == "第一句。第二句[旺柴]。", text)
check("保留非 Plain 组件", len(others) == 1 and isinstance(others[0], Image))
check("空链不炸", p._collect_text([]) == ("", []))

# ------------------------------------------------------------------ 6 音频归一化

print("== 6 音频归一化 ==")
p, _ = new_plugin()
b64 = base64.b64encode(b"hello").decode()
check("bytes", p._normalize_audio(b"hello") == {"base64": b64})
check("base64 字符串", p._normalize_audio(b64) == {"base64": b64})
check("data URI", p._normalize_audio("data:audio/mp3;base64," + b64) == {"base64": b64})
check("dict.audio", p._normalize_audio({"audio": b64}) == {"base64": b64})
check("list 取首项", p._normalize_audio([b64]) == {"base64": b64})
check("对象属性", p._normalize_audio(types.SimpleNamespace(base64=b64)) == {"base64": b64})
check("None", p._normalize_audio(None) is None)
check("普通短串不算音频", p._normalize_audio("oops") is None)
with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as fh:
    fh.write(b"fileaudio")
    tmp = fh.name
try:
    check("文件路径", p._normalize_audio(tmp) == {"base64": base64.b64encode(b"fileaudio").decode()})
finally:
    os.unlink(tmp)

# ------------------------------------------------------------------ 7 意图结论

print("== 7 意图结论 ==")
def decide(plugin, text, umo=U, judge=None, session=False, once=False):
    if judge is not None:
        plugin._judge_llm = lambda e, r, u: asyncio.sleep(0, result=judge)
    plugin._session_on = lambda u: session
    if once:
        plugin._set_once(umo)
    return run(plugin._decide(FakeEvent(text, umo), text, umo))

p, _ = new_plugin(mode="keyword")
check("keyword 命中 → once", decide(p, "用语音跟我说晚安") == {"voice": True, "scope": "once", "off": False})
check("keyword 闲聊 → 不动", decide(p, "今天天气真闷啊")["voice"] is False)
check("关闭意图 → off", decide(p, "还是打字吧")["off"] is True)
check("once 标记消费一次", decide(p, "今天天气真闷啊", once=True)["voice"] is True)
check("once 不重复消费", decide(p, "今天天气真闷啊")["voice"] is False)
check("once 消费后已落盘", not p.state.get("once", {}).get(U))
p._set_once(U)
check("说停时清掉待消费 once", decide(p, "还是打字吧")["voice"] is False
      and not p.state.get("once", {}).get(U))
check("session 模式直接强制", decide(p, "今天天气真闷啊", session=True)["voice"] is True)

p2, _ = new_plugin(mode="llm")
check("llm want=true", decide(p2, "懒得看，你直接讲吧",
                              judge={"want": True, "scope": "once", "off": False})["voice"] is True)
check("llm want=false 放行", decide(p2, "给我讲个故事",
                                    judge={"want": False, "scope": "once", "off": False})["voice"] is False)
check("llm 判失败放行", decide(p2, "懒得看，你直接讲吧", judge=None)["voice"] is False)
check("llm scope=session", decide(p2, "以后都用语音回我吧",
                                  judge={"want": True, "scope": "session", "off": False})["scope"] == "session")
check("llm 非法 scope 归 once", decide(p2, "用语音说",
                                       judge={"want": True, "scope": "WAT", "off": False})["scope"] == "once")
check("llm off 优先", decide(p2, "别发语音了",
                             judge={"want": True, "scope": "session", "off": True})["off"] is True)
check("问功能本身不算要语音（llm 档）",
      decide(p2, "你会发语音吗", judge={"want": False, "scope": "once", "off": False})["voice"] is False)
p2._judge_llm = lambda e, r, u: (_ for _ in ()).throw(RuntimeError("down"))
check("判定时抛异常也放行", run(p2._decide(FakeEvent("用语音说", U), "用语音说", U))["voice"] is False)

# ------------------------------------------------------------------ 8 端到端装饰

print("== 8 端到端装饰 ==")
def decorate(plugin, text, chain=None, synth=None, record=None):
    ev = FakeEvent(text, U, chain if chain is not None else [Plain("晚安，早点睡[旺柴]。")])
    if synth is not None:
        plugin._synthesize = synth
    if record is not None:
        plugin._make_record = record
    run(plugin.on_decorating_result(ev))
    return ev

async def ok_synth(text):
    return {"base64": base64.b64encode(b"AUDIO").decode()}

async def bad_synth(text):
    raise RuntimeError("tts down")

p, _ = new_plugin(mode="keyword")
ev = decorate(p, "用语音说句晚安", synth=ok_synth)
check("命中 → chain 换成 Record", len(ev._result.chain) == 1 and isinstance(ev._result.chain[0], Record),
      ev._result.chain)
check("命中 → 朗读文本已清表情", "[旺柴]" not in str(p._clean_for_tts("晚安，早点睡[旺柴]。")))

p, _ = new_plugin(mode="keyword")
ev = decorate(p, "今天天气真闷啊", synth=ok_synth)
check("未命中 → 一个字不动", len(ev._result.chain) == 1 and isinstance(ev._result.chain[0], Plain),
      ev._result.chain)

p, _ = new_plugin(mode="keyword")
ev = decorate(p, "用语音说", chain=[Record(base64="xxx")], synth=ok_synth)
check("已有 Record → 跳过不二次合成", isinstance(ev._result.chain[0], Record) and len(ev._result.chain) == 1)

p, _ = new_plugin(mode="keyword", max_tts_chars=5)
ev = decorate(p, "用语音说", chain=[Plain("这是一段很长很长的回复内容")], synth=ok_synth)
check("超长 → 保留文字", isinstance(ev._result.chain[0], Plain), ev._result.chain)

p, _ = new_plugin(mode="keyword")
ev = decorate(p, "用语音说", synth=bad_synth)
check("合成失败 → 保住原回复", isinstance(ev._result.chain[0], Plain), ev._result.chain)

p, _ = new_plugin(mode="keyword", attach_text=True)
ev = decorate(p, "用语音说", synth=ok_synth)
check("attach_text → 语音 + 原文", len(ev._result.chain) == 2
      and isinstance(ev._result.chain[0], Record) and isinstance(ev._result.chain[1], Plain),
      ev._result.chain)
check("attach_text 原文保留表情标记", "[旺柴]" in ev._result.chain[1].text, ev._result.chain[1].text)

p, _ = new_plugin(mode="keyword")
ev = decorate(p, "用语音说", chain=[Plain("看图"), Image(), Plain("第二句")], synth=ok_synth)
check("非 Plain 组件保留", any(isinstance(c, Image) for c in ev._result.chain), ev._result.chain)

p, _ = new_plugin(mode="keyword")
ev = decorate(p, "用语音说", chain=[Plain("[NO_REPLY]")], synth=ok_synth)
check("静默暗号不发语音", isinstance(ev._result.chain[0], Plain), ev._result.chain)

p, _ = new_plugin(mode="keyword", enabled=False)
ev = decorate(p, "用语音说", synth=ok_synth)
check("总开关关闭 → 不介入", isinstance(ev._result.chain[0], Plain))

# session 开 → 后续轮强制 → 关闭
p, _ = new_plugin(mode="llm")
p._judge_llm = lambda e, r, u: asyncio.sleep(
    0, result={"want": True, "scope": "session", "off": False})
ev = decorate(p, "以后都用语音回我吧", synth=ok_synth)
check("session 命中 → 转语音", isinstance(ev._result.chain[0], Record), ev._result.chain)
check("session 已写入", p._session_on(U))
p._judge_llm = lambda e, r, u: asyncio.sleep(0, result=None)
ev = decorate(p, "今天天气真闷啊", chain=[Plain("是挺闷的")], synth=ok_synth)
check("session 后续轮不再判定也强制", isinstance(ev._result.chain[0], Record), ev._result.chain)
ev = decorate(p, "还是打字吧", chain=[Plain("好，改回文字")], synth=ok_synth)
check("关闭后 session 清空", not p._session_on(U))
check("关闭本轮保持文字", isinstance(ev._result.chain[0], Plain), ev._result.chain)

# ------------------------------------------------------------------ 9 合成路径

print("== 9 合成路径 ==")
p, ctx = new_plugin()
res = run(p._synthesize("晚安"))
check("优先 context.tts", res == {"base64": base64.b64encode(b"FAKEAUDIO").decode()}, res)
check("context.tts 被调用", len(ctx.tts.calls) == 1)

ctx2 = FakeContext(provider=FakeProvider(payload=base64.b64encode(b"PROVIDER-AUDIO-LONG").decode()))
p2 = mod.VoiceIntentPlugin(ctx2, {})
check("无 context.tts → 退到 provider.get_audio",
      run(p2._synthesize("晚安")) == {"base64": base64.b64encode(b"PROVIDER-AUDIO-LONG").decode()})

ctx3 = FakeContext(provider=FakeProvider(payload=None, boom=True))
p3 = mod.VoiceIntentPlugin(ctx3, {})
try:
    run(p3._synthesize("晚安"))
    check("无可用音频应抛错", False)
except RuntimeError:
    check("无可用音频应抛错", True)

ctx4 = FakeContext(provider=None)
p4 = mod.VoiceIntentPlugin(ctx4, {})
try:
    run(p4._synthesize("晚安"))
    check("无 provider 应抛明确错", False)
except RuntimeError as exc:
    check("无 provider 应抛明确错", "TTS Provider" in str(exc), str(exc))

check("make_record base64", isinstance(p._make_record({"base64": "QUJD"}), Record))
check("make_record file", isinstance(p._make_record({"file": "/tmp/a.mp3"}), Record))

# ------------------------------------------------------------------ 10 指令矩阵

print("== 10 指令矩阵 ==")
p, _ = new_plugin()
matrix = [
    ("/voice", "开口吧 · 状态"),
    ("/voice 状态", "判定档位"),
    ("/voice 开", "接下来我都用语音"),
    ("/voice 关", "改回打字"),
    ("/voice 一次", "下一句我用语音"),
    ("/voice 判定 llm", "已切到 llm"),
    ("/voice 判定", "当前判定档位"),
    ("/voice 清", "都已清空"),
    ("/voice 自检", "结论：全部就绪"),
    ("/voice 帮助", "指令"),
    ("/voice 喵喵", "没认出参数"),
    ("/开口 帮助", "指令"),
    ("状态", "判定档位"),
]
for whole, keyword in matrix:
    out = run(p.cmd_admin(FakeEvent(whole, U), whole))
    check("指令 " + whole, keyword in out, (whole, out))
run(p.cmd_admin(FakeEvent("/voice 开", U), "/voice 开"))
check("开后状态变化", "接下来都用语音" in run(p.cmd_admin(FakeEvent("/voice 状态", U), "/voice 状态")))
run(p.cmd_admin(FakeEvent("/voice 关", U), "/voice 关"))
check("关后状态变化", "普通文字" in run(p.cmd_admin(FakeEvent("/voice 状态", U), "/voice 状态")))

# ------------------------------------------------------------------ 11 模型感知层

print("== 11 模型感知层 ==")

class FakeReq:
    def __init__(self, base="你是大肥鱼。"):
        self.system_prompt = base


def request(plugin, text, req=None, umo=U):
    ev = FakeEvent(text, umo)
    r = req if req is not None else FakeReq()
    run(plugin.on_llm_request(ev, r))
    return r


MARK = mod.AWARE_MARK

p, _ = new_plugin(mode="keyword")
r = request(p, "用语音跟我说句晚安")
check("本轮命中 → 注入感知标记", MARK in r.system_prompt, r.system_prompt)
check("本轮命中 → 明确告知是语音", "**语音**" in r.system_prompt and "说" in r.system_prompt,
      r.system_prompt)
check("本轮命中 → 要求写成可朗读", "markdown" in r.system_prompt)
check("本轮命中 → 保留原人格", r.system_prompt.startswith("你是大肥鱼。"))
check("本轮命中 → 写了 pending", p._take_pending(U) == {"voice": True, "scope": "once", "off": False},
      p.state.get("pending"))

p, _ = new_plugin(mode="keyword")
r = request(p, "今天天气真闷啊")
check("闲聊 → 一个字都不动", r.system_prompt == "你是大肥鱼。", r.system_prompt)

p, _ = new_plugin(mode="keyword")
r0 = FakeReq()
request(p, "用语音说句晚安", r0)
before = r0.system_prompt
request(p, "用语音说句晚安", r0)
check("同轮多次 LLM 请求去重", r0.system_prompt == before and r0.system_prompt.count(MARK) == 1,
      r0.system_prompt.count(MARK))

p, _ = new_plugin(mode="keyword", aware_extra="称呼要带姓氏。")
r = request(p, "用语音说")
check("aware_extra 追加", "称呼要带姓氏。" in r.system_prompt)

p, _ = new_plugin(mode="keyword", in_request_aware=False)
r = request(p, "用语音说")
check("关感知层 → 前置不碰 req", r.system_prompt == "你是大肥鱼。", r.system_prompt)
check("关感知层 → 不写 pending", p._take_pending(U) is None)

class NoSystemReq:
    def __init__(self):
        self.prompt = "hello"


p, _ = new_plugin()
r = request(p, "用语音说", NoSystemReq())
check("req 无 system_prompt → 静默降级不崩", not hasattr(r, "system_prompt"))

# 上一轮回执
p, _ = new_plugin(mode="keyword")
p._note_voice(U, "晚安，早点睡。", False, "once")
check("登记后能读到", p._voice_note(U) is not None)
r = request(p, "今天天气真闷啊")
check("下一轮 → 注入回执", "上一条回复是以" in r.system_prompt and "不是打字" in r.system_prompt,
      r.system_prompt)
check("回执带上实际朗读内容", "晚安，早点睡。" in r.system_prompt, r.system_prompt)
check("回执说明没附文字", "只收到了语音" in r.system_prompt, r.system_prompt)
check("回执标记已注入", p._voice_note(U).get("injected") is True)
r2 = request(p, "再说一句")
check("回执只注入一次", "上一条回复是以" not in r2.system_prompt, r2.system_prompt)

p, _ = new_plugin(mode="keyword", attach_text=True)
p._note_voice(U, "好呀。", True, "once")
r = request(p, "嗯")
check("附原文 → 回执如实说明语音+文字", "语音和文字都收到了" in r.system_prompt, r.system_prompt)

# TTL
p, _ = new_plugin(mode="keyword", receipt_ttl_minutes=1)
p._note_voice(U, "早。", False, "once")
p.state["last_voice"][U]["ts"] = 0
check("回执过期自动清", p._voice_note(U) is None)

# 持续模式：每轮明确告知，且不注入回执
p, _ = new_plugin(mode="keyword")
p._set_session(U, True)
p._note_voice(U, "早。", False, "session")
r = request(p, "今天天气真闷啊")
check("session → 注入持续媒介提示", "一直在**用语音**" in r.system_prompt, r.system_prompt)
check("session → 不重复注入回执", "上一条回复是以" not in r.system_prompt, r.system_prompt)

# 关闭意图：清实况、不告知本轮语音
p, _ = new_plugin(mode="keyword")
p._set_session(U, True)
p._note_voice(U, "早。", False, "session")
r = request(p, "还是打字吧")
check("关闭 → 清掉上一轮实况", p._voice_note(U) is None)
check("关闭 → 不注入本轮语音告知", "你这一轮的回复会被直接转成" not in r.system_prompt,
      r.system_prompt)

# llm / always 档：前置不调模型、不写 pending，只给温和提示
p, ctx = new_plugin(mode="llm")
called = {"n": 0}
async def _fake_judge(e, raw, u):
    called["n"] += 1
    return {"want": True, "scope": "once", "off": False}
p._judge_llm = _fake_judge
r = request(p, "懒得看，你直接讲吧")
check("llm 档前置零模型调用", called["n"] == 0, called)
check("llm 档前置不写 pending", p._take_pending(U) is None)
check("llm 档给温和文风提示", "可能被转成语音" in r.system_prompt, r.system_prompt)

p, _ = new_plugin(mode="always")
r = request(p, "今天天气真闷啊")
check("always 档前置不调模型", MARK in r.system_prompt and "可能被转成语音" in r.system_prompt)

# 端到端：第 N 轮发语音 → 第 N+1 轮模型确实被告知
p, _ = new_plugin(mode="keyword")
ev1 = FakeEvent("用语音说句晚安", U, [Plain("晚安，早点睡[旺柴]。")])
async def _ok(text):
    return {"base64": base64.b64encode(b"AUDIO").decode()}
p._synthesize = _ok
r0 = FakeReq()
run(p.on_llm_request(ev1, r0))
run(p.on_decorating_result(ev1))
check("端到端① 确实发出语音", isinstance(ev1._result.chain[0], Record), ev1._result.chain)
note = p._voice_note(U)
check("端到端② 登记的是朗读版本（无表情标记）",
      note and note["spoken"] == "晚安，早点睡。" and "[旺柴]" not in note["spoken"], note)
ev2 = FakeEvent("你刚才那条语音再说一遍", U, [Plain("好。")])
r1 = FakeReq("你是大肥鱼。")
run(p.on_llm_request(ev2, r1))
check("端到端③ 下一轮模型拿到回执", "晚安，早点睡。" in r1.system_prompt
      and "上一条回复是以" in r1.system_prompt, r1.system_prompt)

# 合成失败绝不能「以为自己说了话」
p, _ = new_plugin(mode="keyword")
async def _bad(text):
    raise RuntimeError("tts down")
p._synthesize = _bad
ev3 = FakeEvent("用语音说", U, [Plain("晚安。")])
run(p.on_decorating_result(ev3))
check("合成失败 → 不登记语音实况", p._voice_note(U) is None, p.state.get("last_voice"))
check("合成失败 → 保住文字", isinstance(ev3._result.chain[0], Plain))

# pending 被 decorating 消费后不残留
p, _ = new_plugin(mode="keyword")
ev4 = FakeEvent("用语音说", U, [Plain("好。")])
p._synthesize = _ok
run(p.on_llm_request(ev4, FakeReq()))
check("前置写了 pending", p.state.get("pending", {}).get(U) is not None)
run(p.on_decorating_result(ev4))
check("decorating 消费后 pending 清空", p.state.get("pending", {}).get(U) is None)
check("pending 消费后不重复判", isinstance(ev4._result.chain[0], Record))

# 指令
p, _ = new_plugin()
out = run(p.cmd_admin(FakeEvent("/voice 感知", U), "/voice 感知"))
check("指令 /感知 无实况", "无（最近没发过语音" in out, out)
p._note_voice(U, "早。", False, "once")
out = run(p.cmd_admin(FakeEvent("/voice 感知", U), "/voice 感知"))
check("指令 /感知 有实况", "上一轮：语音" in out and "早。" in out, out)
out = run(p.cmd_admin(FakeEvent("/voice 自检", U), "/voice 自检"))
check("自检含感知钩子", "on_llm_request" in out and "_note_voice" in out and "全部就绪" in out, out)
out = run(p.cmd_admin(FakeEvent("/voice 清", U), "/voice 清"))
check("清空也清实况", p._voice_note(U) is None)

# ------------------------------------------------------------------ 12 全量指令：解析 + 生效

print("== 12 全量指令：解析 + 生效 ==")

MATRIX = mod.COMMAND_MATRIX
check("指令矩阵非空", len(MATRIX) >= 30, len(MATRIX))
check("手动别名全部登记", len(mod.MANUAL_HEADS) >= 8, mod.MANUAL_HEADS)

# 12.1 解析层：矩阵里每一条都逐条实跑 _parse_cmd
parse_bad = []
for text, want_head, want_rest, kind in MATRIX:
    h, r = mod.VoiceIntentPlugin._parse_cmd(text)
    if h != want_head.lower() or (want_rest and r != want_rest) or h not in mod.ALL_KNOWN_HEADS:
        parse_bad.append((text, h, r, want_head, want_rest))
check(f"解析层 {len(MATRIX)}/{len(MATRIX)} 全通过", not parse_bad, parse_bad[:5])

# 12.2 生效层：每条指令都用全新插件实跑 cmd_admin，断言不是「没认出参数」+ 副作用正确
def effects(head, p):
    return {
        "session": p._session_on(U),
        "once": bool(p.state.get("once", {}).get(U)),
        "mode": p._mode(),
        "state_empty": (not p.state.get("sessions") and not p.state.get("once")
                        and p.state.get("mode_override") is None),
    }

run_bad = []
for text, want_head, want_rest, kind in MATRIX:
    p, ctx = new_plugin(mode="keyword")
    if kind == "manual":
        async def _fake_syn(t):
            return {"base64": base64.b64encode(b"AUDIO").decode()}
        p._synthesize = _fake_syn
    ev = FakeEvent(text, U)
    out = run(p.cmd_admin(ev, text))
    if isinstance(out, dict):
        ok = (out.get("__voice__") is True
              and isinstance(out.get("chain"), list)
              and isinstance(out["chain"][0], Record)
              and out.get("spoken") == want_rest.replace("[旺柴]", ""))
        if kind != "manual":
            ok = False   # 管理指令不该返回语音链
        note = p._voice_note(U)
        if ok and (not note or note.get("by") != "command" or note.get("spoken") != want_rest):
            ok = False
    else:
        ok = isinstance(out, str) and "没认出参数" not in out and len(out) > 0
        if ok:
            h = want_head.lower()
            e = effects(h, p)
            if h in ("开", "on", "开启"):
                ok = e["session"] is True
            elif h in ("关", "off", "关闭", "停", "stop", "清", "reset", "清空"):
                ok = e["session"] is False
            elif h in ("一次", "once", "这轮"):
                ok = e["once"] is True
            elif h in ("判定", "mode", "档位") and want_rest in ("keyword", "llm", "always"):
                ok = e["mode"] == want_rest
            elif h in ("校验", "verify", "验指令", "检查指令"):
                ok = "全部通过" in out and f"{len(MATRIX)}/{len(MATRIX)}" in out
            elif h in ("帮助", "help", "指令", "commands"):
                ok = "/开口 发" in out and "/开口 校验" in out
            elif h in ("自检", "selftest"):
                ok = "全部就绪" in out
            elif h in ("感知", "aware", "记性"):
                ok = "模型感知层" in out
            elif h in ("", "状态", "status", "检查", "check"):
                ok = "状态" in out
    if not ok:
        run_bad.append((text, type(out).__name__, str(out)[:90]))
check(f"生效层 {len(MATRIX)}/{len(MATRIX)} 全通过（含状态副作用）", not run_bad, run_bad[:5])

# 12.3 手动指令所有别名都要真的合成并 yield 语音链
alias_bad = []
for head in mod.MANUAL_HEADS:
    p, _ = new_plugin()
    async def _fake_syn(t):
        _fake_syn.got = t
        return {"base64": base64.b64encode(b"AUDIO").decode()}
    p._synthesize = _fake_syn
    text = f"/voice {head} 晚安"
    collected = []
    async def _collect():
        async for item in p._run_command(FakeEvent(text, U)):
            collected.append(item)
    run(_collect())
    ok = (len(collected) == 1 and isinstance(collected[0], tuple)
          and collected[0][0] == "chain"
          and isinstance(collected[0][1][0], Record)
          and getattr(_fake_syn, "got", "") == "晚安")
    if not ok:
        alias_bad.append((head, collected, getattr(_fake_syn, "got", None)))
check(f"手动 {len(mod.MANUAL_HEADS)} 个别名逐条真发语音", not alias_bad, alias_bad[:4])

# 12.4 手动指令的边界：空内容 / 超长 / 合成失败，都要给明确文字而非静默
p, _ = new_plugin()
out = run(p.cmd_admin(FakeEvent("/voice 发", U), "/voice 发"))
check("手动空内容 → 明确提示不合成", isinstance(out, str) and "要念什么" in out, out)
check("手动空内容 → 不登记实况", p._voice_note(U) is None)

p, _ = new_plugin(max_tts_chars=5)
out = run(p.cmd_admin(FakeEvent("/voice 发 今晚早点睡啦", U), "/voice 发 今晚早点睡啦"))
check("手动超长 → 明确提示不合成", isinstance(out, str) and "超过朗读上限" in out, out)

p, _ = new_plugin()
async def _boom(t):
    raise RuntimeError("tts down")
p._synthesize = _boom
out = run(p.cmd_admin(FakeEvent("/voice 发 晚安", U), "/voice 发 晚安"))
check("手动合成失败 → 明确回话", isinstance(out, str) and "语音合成失败" in out and "晚安" in out, out)
check("手动合成失败 → 不登记实况", p._voice_note(U) is None)

# 12.5 手动指令不会被装饰层二次处理（链里已有 Record）
p, _ = new_plugin()
async def _ok(t):
    return {"base64": base64.b64encode(b"AUDIO").decode()}
p._synthesize = _ok
out = run(p.cmd_admin(FakeEvent("/voice 发 晚安", U), "/voice 发 晚安"))
ev = FakeEvent("/voice 发 晚安", U, out["chain"])
run(p.on_decorating_result(ev))
check("手动语音不被二次合成", len(ev._result.chain) == 1
      and isinstance(ev._result.chain[0], Record), ev._result.chain)

# 12.6 感知层：指令发的语音，回执明确「不是你生成的」
p, _ = new_plugin(mode="keyword")
p._synthesize = _ok
run(p.cmd_admin(FakeEvent("/voice 发 晚安", U), "/voice 发 晚安"))
r = FakeReq()
run(p.on_llm_request(FakeEvent("刚才那条语音说啥", U), r))
check("指令语音 → 回执说明由指令发出", "由指令直接" in r.system_prompt, r.system_prompt)
check("指令语音 → 明确不是模型生成", "内容不是你生成的" in r.system_prompt, r.system_prompt)
check("指令语音 → 仍带上朗读内容", "晚安" in r.system_prompt)
out = run(p.cmd_admin(FakeEvent("/voice 感知", U), "/voice 感知"))
check("/感知 显示 command 来源", "上一轮：语音" in out, out)

# 12.7 运行时 /开口 校验 输出与矩阵一致
p, _ = new_plugin()
out = run(p.cmd_admin(FakeEvent("/voice 校验", U), "/voice 校验"))
check("/校验 报全部通过", "全部通过 ✅" in out and f"{len(MATRIX)}/{len(MATRIX)}" in out, out)
check("/校验 声明零状态改动", "零状态改动" in out, out)
check("/校验 不改会话状态", not p._session_on(U) and not p.state.get("once", {}).get(U))

# 12.8 帮助清单 / README / 矩阵三者一致（文档写了的，代码必须认）
help_out = run(p.cmd_admin(FakeEvent("/voice 帮助", U), "/voice 帮助"))
for key in ("发 <内容>", "校验", "自检", "感知", "判定", "一次", "清", "帮助"):
    check(f"帮助清单含「{key}」", key in help_out, help_out)
readme = open(os.path.join(HERE, "README.md"), encoding="utf-8").read()
for key in ("/开口 发", "/开口 校验", "/开口 感知", "/开口 一次", "/voice say"):
    check(f"README 含「{key}」", key in readme, key)
check("README 指令表与运行时清单同源", "COMMAND_MATRIX" in readme)

# ------------------------------------------------------------------ 13 At 段污染回归

print("== 13 At 段污染回归（QQ 聊天截图复现） ==")

# 用户连发 3 次 /开口 校验，机器人回「没认出参数「[主人(2581793700)]」」。
# 根因：At 组件被渲染成 [主人(QQ)] 拼进 message_str，_parse_cmd 把它当指令名。
AT_FORMS = [
    "[主人(2581793700)]/开口 校验",
    "[主人(2581793700)] 校验",
    "/开口 校验[主人(2581793700)]",
    "[CQ:at,qq=2581793700] /开口 自检",
    "／开口 感知",
    "/开口\u200b自检",
    "@机器人 /voice 帮助",
    "@机器人(2581793700) /voice 发 晚安",
]
for t in AT_FORMS:
    h, _ = mod.VoiceIntentPlugin._parse_cmd(t)
    check(f"At 污染形态解析：{t[:24]}", h in mod.ALL_KNOWN_HEADS, (t, h))

# 逐条实跑 cmd_admin，绝不能掉进「没认出参数」
p, _ = new_plugin()
async def _fake_syn(t):
    return {"base64": base64.b64encode(b"AUDIO").decode()}
p._synthesize = _fake_syn
for t in AT_FORMS:
    out = run(p.cmd_admin(FakeEvent(t, U), t))
    ok = (not isinstance(out, str)) or ("没认出参数" not in out)
    check(f"At 污染形态不报没认出参数：{t[:24]}", ok, out)

# 关键：清洗不能吃掉 face_convert 的表情标记
check("清洗保留 [旺柴]", mod.VoiceIntentPlugin._clean_input("晚安[旺柴]") == "晚安[旺柴]",
      mod.VoiceIntentPlugin._clean_input("晚安[旺柴]"))
check("清洗保留 [微笑]", "[微笑]" in mod.VoiceIntentPlugin._clean_input("用语音说[微笑]"))
check("清洗确实删掉 [主人(QQ)]",
      mod.VoiceIntentPlugin._clean_input("[主人(2581793700)]/开口 校验") == "/开口 校验")

# 多个 At 前缀连着出现也要清干净
check("多个 At 前缀",
      mod.VoiceIntentPlugin._parse_cmd("@机器人 @大肥鱼 /voice 自检")[0] == "自检",
      mod.VoiceIntentPlugin._parse_cmd("@机器人 @大肥鱼 /voice 自检"))

# _plain_from_event：跳过 At 组件，只拼 Plain
class _At:
    def __init__(self, qq):
        self.qq = qq
class _MsgObj:
    def __init__(self, chain):
        self.message = chain

class _EvWithObj(FakeEvent):
    def __init__(self, text, chain):
        super().__init__(text, U)
        self.message_obj = _MsgObj(chain)

plain = mod.VoiceIntentPlugin._plain_from_event(_EvWithObj("[主人(2581793700)]", [
    _At(2581793700), Plain("/开口 校验"),
]))
check("消息链跳过 At 只取 Plain", plain == "/开口 校验", plain)

# _raw_text：message_str 只剩 At 时，能从消息链里把真指令捞回来
ev = _EvWithObj("[主人(2581793700)]", [_At(2581793700), Plain("/开口 校验")])
raw = mod.VoiceIntentPlugin._raw_text(ev, "[主人(2581793700)]")
check("_raw_text 从消息链捞回真指令", "校验" in raw, raw)
h, _ = mod.VoiceIntentPlugin._parse_cmd(raw)
check("_raw_text 结果可解析为校验", h == "校验", h)

# 端到端：模拟框架只给 At 文本，_run_command 仍应返回校验结果
p2, _ = new_plugin()
collected = []
async def _collect2():
    async for item in p2._run_command(ev):
        collected.append(item)
run(_collect2())
check("_run_command 端到端不报没认出参数",
      any(isinstance(x, str) and "全部通过" in x for x in collected), collected)

# 诊断行：真遇到未知参数时，能看到清洗后原文
p3, _ = new_plugin()
out = run(p3.cmd_admin(FakeEvent("/voice 瞎写的东西", U), "/voice 瞎写的东西"))
check("兜底带 [诊断] 原文", "[诊断]" in out and "瞎写的东西" in out, out)

# ------------------------------------------------------------------ 结果

print()
if FAILED:
    print(f"结果：{len(FAILED)} 项失败 / 共 {len(PASSED) + len(FAILED)} 项")
    for name, detail in FAILED:
        print("  -", name, detail)
    sys.exit(1)
print(f"结果：全部通过 OK（{len(PASSED)} 项）")
