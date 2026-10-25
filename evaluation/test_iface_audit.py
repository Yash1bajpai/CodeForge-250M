from iface_audit import *
T=lambda s: target_of('"""\nWrite x\nYour code should pass these tests:\n\n'+s+'\n"""\n')
assert target_of('def double(n):\n  pass\n"""\n[link](https:// write a function.\nYour code should pass these tests:\n\nassert s(1)==2\n"""\n')==('s',1)  # corrupted task text (MBPP 118 style)
def t(code,tgt): return classify(code,tgt)
# fixtures
assert T('assert f(1,2)==3\nassert f(0,0)==0')==('f',2)
assert T('assert f(1,2)==3\nassert f(1)==3') is None           # multi-arity callsites
assert T('assert f(*a)==3') is None and T('assert f(1,x=2)==3') is None
assert T('assert g(f(1))==1')==('g',1)                          # outer call is target
assert T('assert 1==1') is None
assert t('def f(a,b): return 1',('f',2))=='arity_ok'
assert t('def f(a): return 1',('f',2))=='arity_wrong'
assert t('def f(a,b=1): return 1',('f',1))=='arity_ok' and t('def f(a,b=1): return 1',('f',2))=='arity_ok'
assert t('def f(a,b=1): return 1',('f',3))=='arity_wrong'
assert t('def f(*a): pass',('f',2))=='unknown_def' and t('def f(**k): pass',('f',2))=='unknown_def'
assert t('async def f(a): pass',('f',1))=='unknown_def'
assert t('@d\ndef f(a): pass',('f',1))=='unknown_def'
assert t('def f(a,*,b): pass',('f',1))=='arity_wrong'            # required kw-only
assert t('def f(a,*,b=1): pass',('f',1))=='arity_ok'
assert t('def f(n,n): pass',('f',2))=='invalid_def'
assert t('def f(a): pass\ndef f(a,b): pass',('f',2))=='arity_ok'  # last wins
assert t('def f(a,b): pass\ndef f(a): pass',('f',2))=='arity_wrong'
assert t('class C:\n    def f(self,a): pass',('f',1))=='nodef'    # methods not top-level
assert t('def g():\n    def f(a): pass',('f',1))=='name_missing'  # nested not callable as global
assert t('x=1',('f',1))=='nodef' and t('def f(:',('f',1))=='noparse'
assert t('def f(a): pass',None)=='unknown_target'
print('all fixtures pass')
