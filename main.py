from __future__ import division
import re
import sys
import io
import math
import operator as op

Symbol = str          # A Lisp Symbol is implemented as a Python str
List = list         # A Lisp List is implemented as a Python list
# A Lisp Number is implemented as a Python int, only integer is allowed in our language
Number = int


mse = ["note", "sequence", "seqn-p", "seqn-v", "seqn-d", "seq-append", "with", "fun", "interleave", "insert", "transpose",
       "changePits", "changeVels", "changeDurs", "retrograde", "zip", "markov"]
dmse = ["note", "sequence", "seqn", "app", "fun", "insert",
        "transpose", "changeProp", "retrograde", "zip", "markov"]


def parse(program):
    "Read a Scheme expression from a string."
    return read_from_tokens(tokenize(program))


def tokenize(s):
    "Convert a string into a list of tokens."
    return s.replace('(', ' ( ').replace(')', ' ) ').split()


def read_from_tokens(tokens):
    "Read an expression from a sequence of tokens."
    if len(tokens) == 0:
        raise SyntaxError('unexpected EOF while reading')
    token = tokens.pop(0)
    if '(' == token:
        L = []
        while tokens[0] != ')':
            L.append(read_from_tokens(tokens))
        tokens.pop(0)  # pop off ')'
        return L
    elif ')' == token:
        raise SyntaxError('unexpected )')
    else:
        return atom(token)


def atom(token):
    "Numbers become numbers; every other token is a symbol."
    try:
        return int(token)
    except ValueError:
        return Symbol(token)


def make_seq(x):
    seq = ["sequence"]
    for item in x:
        if isinstance(item, note):
            if item.is_valid_note():
                seq.append(item)
    return seq


def standard_env():
    "An environment with some Scheme standard procedures and our pre-defined functions."
    env = Env()
    env.update(vars(math))  # sin, cos, sqrt, pi, ...
    env.update({
        '+': op.add, '-': op.sub, '*': op.mul, '/': op.truediv,
        '>': op.gt, '<': op.lt, '>=': op.ge, '<=': op.le, '=': op.eq,
        'note': lambda pit, vel, dur: note(pit, vel, dur),
        'seqn': lambda prop, nums: seqn(nums,prop, env),
        'interleave': lambda seq1, seq2: interleave(seq1, seq2),
        'insert': lambda seq1, seq2, index: seq_insert(seq1,seq2,index),
        'transpose': lambda seq, val: transpose(seq,val),
        'changeProp': lambda prop, seq, val: changeProp(prop,seq,val),
        'retrograde': lambda seq: retro(seq),
        'app': lambda proc, args: proc(*args),
        'begin': lambda *x: x[-1],
        'car': lambda x: x[0],
        'cdr': lambda x: x[1:],
        'cons': lambda x, y: [x] + y,
        'eq?':     op.is_,
        'equal?':  op.eq,
        'length':  len,
        'list?': lambda x: isinstance(x, list),
        'map':     map,
        'max':     max,
        'min':     min,
        'not':     op.not_,
        'null?': lambda x: x == [],
        'number?': lambda x: isinstance(x, Number),
        'procedure?': callable,
        'round':   round,
        'symbol?': lambda x: isinstance(x, Symbol),
        'default_pitch': 30,
        'default_vel': 30,
        'default_dur': 30
    })
    return env


class Env(dict):
    "An environment: a dict of {'var':val} pairs, with an outer Env."

    def __init__(self, parms=(), args=(), outer=None):
        self.update(zip(parms, args))
        self.outer = outer

    def find(self, var):
        "Find the innermost Env where var appears."
        return self if (var in self) else self.outer.find(var)


global_env = standard_env()


def repl(prompt='lis.py> '):
    "A prompt-read-eval-print loop."
    while True:
        val = eval(parse(input(prompt)))
        if val is not None:
            print(lispstr(val))


def lispstr(exp):
    "Convert a Python object back into a Lisp-readable string."
    if isinstance(exp, List):
        return '(' + ' '.join(map(lispstr, exp)) + ')'
    else:
        return str(exp)


class Procedure(object):
    "A user-defined Scheme procedure."

    def __init__(self, parms, body, env):
        self.parms, self.body, self.env = parms, body, env

    def __call__(self, *args):
        return eval(self.body, Env(self.parms, args, self.env))


def desugar(inpexp):
    "desugar parsing MSE to expressions accepted in eval"
    if len(inpexp) == 0:
        return []
    x = inpexp[0]
    if isinstance(x, int):
        return x
    elif isinstance(x, Symbol):
        return x
    elif isinstance(x, list):
        if len(x) == 0:
            return []
        elif x[0] == "note":
            if len(x) == 4:
                exp = ["note"]
                exp.append(desugar(x[1]))
                exp.append(desugar(x[2]))
                exp.append(desugar(x[3]))
                inpexp.pop(0)
                return exp.append(desugar(inpexp))
            else:
                raise SyntaxError('unexpected # of expressions for note')
        elif x[0] == "sequence":
           exp = ["sequence"]
           x.pop(0)
           inpexp.pop(0)
           exp.extend(desugar(x))
           return exp.append(desugar(inpexp))
        elif x[0] == "seqn-p":
           exp = ["seqn", "p"]
           x.pop(0)
           inpexp.pop(0)
           exp.extend(desugar(x))
           return exp.append(desugar(inpexp))
        elif x[0] == "seqn-v":
           exp = ["seqn", "v"]
           x.pop(0)
           inpexp.pop(0)
           exp.extend(desugar(x))
           return exp.append(desugar(inpexp))
        elif x[0] == "seqn-d":
           exp = ["seqn", "d"]
           x.pop(0)
           inpexp.pop(0)
           exp.extend(desugar(x))
           return exp.append(desugar(inpexp))
        elif x[0] == "seq-append":
           exp = ["insert"]
           x.pop(0)
           inpexp.pop(0)
           exp.extend(x)
           exp.append(-1)
           return exp.append(desugar(inpexp))
        elif x[0] == "with":
           if len(x) == 4:
              (_, ide, nameexpr, body) = x
              funexp = ["fun", ide, body]
              exp = ["app", funexp, nameexpr]
              inpexp.pop(0)
              return desugar(exp).append(desugar(inpexp))
           else:
               raise SyntaxError('unexpected # of expressions for with')
        elif x[0] == "insert": 
            if len(x) == 4:
                exp = ["insert"]
                exp.append(desugar(x[1]))
                exp.append(desugar(x[2]))
                exp.append(desugar(x[3]))
                inpexp.pop(0)
                return exp.append(desugar(inpexp))
            else:
                raise SyntaxError('unexpected # of expressions for insert')
        elif x[0] == "fun":
            if len(x) == 3: 
                exp = ["fun"]
                exp.append(desugar(x[1]))
                exp.append(desugar(x[2]))
                inpexp.pop(0)
                return exp.append(desugar(inpexp))
            else:
                raise SyntaxError('unexpected # of expressions for fun')
        elif x[0] == "app":
            if len(x) == 3:
                exp = ["app"]
                exp.append(desugar(x[1]))
                exp.append(desugar(x[2]))
                inpexp.pop(0)
                return exp.append(desugar(inpexp))
            else:
                raise SyntaxError('unexpected # of expressions for app')
        elif x[0] == "transpose":
            if len(x) == 3:
                exp = ["transpose"]
                exp.append(desugar(x[1]))
                exp.append(desugar(x[2]))
                inpexp.pop(0)
                return exp.append(desugar(inpexp))
            else:
                raise SyntaxError('unexpected # of expressions for transpose')
        elif x[0] == "changePits":
            if len(x) == 3:
                exp == ["changeProp", "p"]
                exp.append(desugar(x[1]))
                exp.append(desugar(x[2]))
                inpexp.pop(0)
                return exp.append(desugar(inpexp))
            else:
                raise SyntaxError('unexpected # of expressions for changePits')
        elif x[0] == "changeVels":
            if len(x) == 3:
                exp == ["changeProp", "v"]
                exp.append(desugar(x[1]))
                exp.append(desugar(x[2]))
                inpexp.pop(0)
                return exp.append(desugar(inpexp))
            else:
                raise SyntaxError('unexpected # of expressions for changeVels')
        elif x[0] == "changeDurs":
            if len(x) == 3:
                exp == ["changeProp", "d"]
                exp.append(desugar(x[1]))
                exp.append(desugar(x[2]))
                inpexp.pop(0)
                return exp.append(desugar(inpexp))
            else:
                raise SyntaxError('unexpected # of expressions for changeDurs')
        elif x[0] == "retrograde":
            if len(x) == 2:
                exp == ["retrograde"]
                exp.append(desugar(x[1]))
                inpexp.pop(0)
                return exp.append(desugar(inpexp))
            else:
                raise SyntaxError('unexpected # of expressions for retrograde')


def eval(x, env=global_env):
    "Evaluate an expression in an environment."
    if isinstance(x, Symbol):      # variable reference
        return env.find(x)[x]
    elif not isinstance(x, List):  # constant literal
        return x
    elif x[0] == 'quote':          # (quote exp)
        (_, exp) = x
        return exp
    elif x[0] == 'if':             # (if test conseq alt)
        (_, test, conseq, alt) = x
        exp = (conseq if eval(test, env) else alt)
        return eval(exp, env)
    elif x[0] == 'define':         # (define var exp)
        (_, var, exp) = x
        env[var] = eval(exp, env)
    elif x[0] == 'set!':           # (set! var exp)
        (_, var, exp) = x
        env.find(var)[var] = eval(exp, env)
    elif x[0] == 'lambda':         # (lambda (var...) body)
        (_, parms, body) = x
        return Procedure(parms, body, env)
    else:                          # (proc arg...)
        proc = eval(x[0], env)
        args = [eval(exp, env) for exp in x[1:]]
        return proc(*args)

def seqn(vals, sym, env):
    if sym == "p":
        newseq = ["sequence"]
        for pitch in vals:
            nextnote = note(pitch,env['default_vel'],env['default_dur'])
            if nextnote.is_valid_pit():
                newseq.append(nextnote)
            else:
                raise SyntaxError('invalid pitch given as' + pitch)
    if sym == 'v':
        newseq = ["sequence"]
        for vel in vals:
            nextnote = note(env['default_pitch'],vel,env['default_dur'])
            if nextnote.is_valid_note():
                newseq.append(nextnote)
            else:
                raise SyntaxError('invalid velocity given as' + vel)
    if sym == 'd':
        newseq = ["sequence"]
        for dur in vals:
            nextnote = note(env['default_pitch'],env['default_vel'],dur)
            if nextnote.is_valid_note():
                newseq.append(nextnote)
            else:
                raise SyntaxError('invalid duration given as' + dur)
    else:
        raise SyntaxError('illegal operation')

def seq_insert(seq1, seq2, pos):
    if seq1[0] == "sequence" and seq2[0] == "sequence":
        if pos > len(seq2):
            seq1.pop(0)
            seq2.extend(seq1)
            return seq2
        elif pos <=0:
            seq2.pop(0)
            seq1.extend(seq2)
            return seq1
        else:
            n = pos
            for notes in seq1:
                seq2.insert(n,notes)
                n= n+1
            return seq2
    else:
        raise SyntaxError('need sequences to do insert')
    
def interleave(seq1, seq2):
    if seq1[0] == "sequence" and seq2[0] == "sequence":
        newseq = ["sequence"]
        seq1.pop(0)
        seq2.pop(0)
        if len(seq1) >= len(seq2):
            for notes in seq2:
                newseq.append(seq1[0])
                newseq.append(notes)
                seq1.pop(0)
            newseq.extend(seq1)
        else:
            for notes in seq1:
                newseq.append(notes)
                newseq.append(seq2[0])               
                seq2.pop(0)
            newseq.extend(seq2)
        return newseq
    else:
        raise SyntaxError('need sequences to do interleave')

def transpose(seq, val):
    if seq[0] == "sequence":
        seq.pop(0)
        for notes in seq:
            notes.set_pit(notes.get_pit()+val)
        seq.insert(0, "sequence")
        return seq
    else:
        raise SyntaxError('need sequences to do transpose')

def changeProp(sym, seq, val):
    if sym == "p" and seq[0] == "sequence":
        newseq = ["sequence"]
        for notes in seq:
            notes.set_pit(val)
            if notes.is_valid_pit():
                newseq.append(notes)
            else:
                raise SyntaxError('invalid pitch given as' + val)
    if sym == 'v' and seq[0] == "sequence":
        newseq = ["sequence"]
        for notes in seq:
            notes.set_vel(val)
            if notes.is_valid_note():
                newseq.append(notes)
            else:
                raise SyntaxError('invalid velocity given as' +val)
    if sym == 'd'and seq[0] == "sequence":
        newseq = ["sequence"]
        for notes in seq:
            notes.set_dur(val)
            if notes.is_valid_note():
                newseq.append(notes)
            else:
                raise SyntaxError('invalid duration given as' + val)
    else:
        raise SyntaxError('illegal operation')

def retro(seq):
    if seq[0] == "sequence":
        seq.pop(0)
        seq.reverse() 
        seq.insert(0,"sequence")
    else:
        raise SyntaxError('need sequences to do retrograde')

class note:
    def __init__(self, pitch, vel, dur):
        self.pitch = pitch
        self.vel = vel
        self.dur = dur

    def is_valid_pit(self):
        if isinstance(self.pitch, int) and self.pitch >= 0 and self.pitch <= 127:
            return True
        elif re.fullmatch('[A-G](#|b)*[0-9]+$', self.pitch):
            return True
        else:
            return False

    def is_valid_note(self):
        if self.is_valid_pit() and isinstance(self.vel, int) and isinstance(self.dur, int):
            return True
        else:
            return False

    def get_pit(self):
        return self.pitch

    def get_vel(self):
        return self.vel

    def get_dur(self):
        return self.dur

    def print_note(self):
        print("pitch: " + self.pitch + " velocity: " +
              self.vel + " duration: " + self.dur)

    def set_pit(self, newpit):
        self.pitch = newpit

    def set_vel(self, newvel):
        self.vel = newvel

    def set_dur(self, newdur):
        self.dur = newdur


class closureV:
    def __init__(self, param, body):
        self.param = param
        self.body = body

    def get_param(self):
        return self.param

    def get_body(self):
        return self.body
