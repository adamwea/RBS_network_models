/* Created by Language version: 7.7.0 */
/* VECTORIZED */
#define NRN_VECTORIZED 1
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "mech_api.h"
#undef PI
#define nil 0
#include "md1redef.h"
#include "section.h"
#include "nrniv_mf.h"
#include "md2redef.h"
 
#if METHOD3
extern int _method3;
#endif

#if !NRNGPU
#undef exp
#define exp hoc_Exp
extern double hoc_Exp(double);
#endif
 
#include "nmodlmutex.h" 
#define nrn_init _nrn_init__GABAa
#define _nrn_initial _nrn_initial__GABAa
#define nrn_cur _nrn_cur__GABAa
#define _nrn_current _nrn_current__GABAa
#define nrn_jacob _nrn_jacob__GABAa
#define nrn_state _nrn_state__GABAa
#define _net_receive _net_receive__GABAa 
#define release release__GABAa 
 
#define _threadargscomma_ _p, _ppvar, _thread, _nt,
#define _threadargsprotocomma_ double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt,
#define _threadargs_ _p, _ppvar, _thread, _nt
#define _threadargsproto_ double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt
 	/*SUPPRESS 761*/
	/*SUPPRESS 762*/
	/*SUPPRESS 763*/
	/*SUPPRESS 765*/
	 extern double *getarg();
 /* Thread safe. No static _p or _ppvar. */
 
#define t _nt->_t
#define dt _nt->_dt
#define Erev _p[0]
#define Erev_columnindex 0
#define fflag _p[1]
#define fflag_columnindex 1
#define sid _p[2]
#define sid_columnindex 2
#define cid _p[3]
#define cid_columnindex 3
#define i _p[4]
#define i_columnindex 4
#define g _p[5]
#define g_columnindex 5
#define Ron _p[6]
#define Ron_columnindex 6
#define Roff _p[7]
#define Roff_columnindex 7
#define synon _p[8]
#define synon_columnindex 8
#define DRon _p[9]
#define DRon_columnindex 9
#define DRoff _p[10]
#define DRoff_columnindex 10
#define v _p[11]
#define v_columnindex 11
#define _g _p[12]
#define _g_columnindex 12
#define _tsav _p[13]
#define _tsav_columnindex 13
#define _nd_area  *_ppvar[0]._pval
 
#if MAC
#if !defined(v)
#define v _mlhv
#endif
#if !defined(h)
#define h _mlhh
#endif
#endif
 
#if defined(__cplusplus)
extern "C" {
#endif
 static int hoc_nrnpointerindex =  -1;
 static Datum* _extcall_thread;
 static Prop* _extcall_prop;
 /* external NEURON variables */
 /* declaration of user functions */
 static double _hoc_Exp1(void*);
 static int _mechtype;
extern void _nrn_cacheloop_reg(int, int);
extern void hoc_register_prop_size(int, int, int);
extern void hoc_register_limits(int, HocParmLimits*);
extern void hoc_register_units(int, HocParmUnits*);
extern void nrn_promote(Prop*, int, int);
extern Memb_func* memb_func;
 
#define NMODL_TEXT 1
#if NMODL_TEXT
static const char* nmodl_file_text;
static const char* nmodl_filename;
extern void hoc_reg_nmodl_text(int, const char*);
extern void hoc_reg_nmodl_filename(int, const char*);
#endif

 extern Prop* nrn_point_prop_;
 static int _pointtype;
 static void* _hoc_create_pnt(Object* _ho) { void* create_point_process(int, Object*);
 return create_point_process(_pointtype, _ho);
}
 static void _hoc_destroy_pnt(void*);
 static double _hoc_loc_pnt(void* _vptr) {double loc_point_process(int, void*);
 return loc_point_process(_pointtype, _vptr);
}
 static double _hoc_has_loc(void* _vptr) {double has_loc_point(void*);
 return has_loc_point(_vptr);
}
 static double _hoc_get_loc_pnt(void* _vptr) {
 double get_loc_point_process(void*); return (get_loc_point_process(_vptr));
}
 extern void _nrn_setdata_reg(int, void(*)(Prop*));
 static void _setdata(Prop* _prop) {
 _extcall_prop = _prop;
 }
 static void _hoc_setdata(void* _vptr) { Prop* _prop;
 _prop = ((Point_process*)_vptr)->_prop;
   _setdata(_prop);
 }
 /* connect user functions to hoc names */
 static VoidFunc hoc_intfunc[] = {
 0,0
};
 static Member_func _member_func[] = {
 "loc", _hoc_loc_pnt,
 "has_loc", _hoc_has_loc,
 "get_loc", _hoc_get_loc_pnt,
 "Exp1", _hoc_Exp1,
 0, 0
};
#define Exp1 Exp1_GABAa
 extern double Exp1( _threadargsprotocomma_ double );
 /* declare global and static user variables */
#define Alpha Alpha_GABAa
 double Alpha = 1;
#define Beta Beta_GABAa
 double Beta = 0.5;
#define Cdur Cdur_GABAa
 double Cdur = 1.08;
#define Rtau Rtau_GABAa
 double Rtau = 0;
#define Rinf Rinf_GABAa
 double Rinf = 0;
#define gmax gmax_GABAa
 double gmax = 1;
 /* some parameters have upper and lower limits */
 static HocParmLimits _hoc_parm_limits[] = {
 0,0,0
};
 static HocParmUnits _hoc_parm_units[] = {
 "Cdur_GABAa", "ms",
 "Alpha_GABAa", "/ms",
 "Beta_GABAa", "/ms",
 "gmax_GABAa", "S",
 "Rtau_GABAa", "ms",
 "Erev", "mV",
 "sid", "1",
 "cid", "1",
 "i", "nA",
 "g", "umho",
 0,0
};
 static double Roff0 = 0;
 static double Ron0 = 0;
 static double delta_t = 1;
 /* connect global user variables to hoc */
 static DoubScal hoc_scdoub[] = {
 "Cdur_GABAa", &Cdur_GABAa,
 "Alpha_GABAa", &Alpha_GABAa,
 "Beta_GABAa", &Beta_GABAa,
 "gmax_GABAa", &gmax_GABAa,
 "Rinf_GABAa", &Rinf_GABAa,
 "Rtau_GABAa", &Rtau_GABAa,
 0,0
};
 static DoubVec hoc_vdoub[] = {
 0,0,0
};
 static double _sav_indep;
 static void nrn_alloc(Prop*);
static void  nrn_init(NrnThread*, _Memb_list*, int);
static void nrn_state(NrnThread*, _Memb_list*, int);
 static void nrn_cur(NrnThread*, _Memb_list*, int);
static void  nrn_jacob(NrnThread*, _Memb_list*, int);
 static void _hoc_destroy_pnt(void* _vptr) {
   destroy_point_process(_vptr);
}
 
static int _ode_count(int);
static void _ode_map(int, double**, double**, double*, Datum*, double*, int);
static void _ode_spec(NrnThread*, _Memb_list*, int);
static void _ode_matsol(NrnThread*, _Memb_list*, int);
 
#define _cvode_ieq _ppvar[3]._i
 static void _ode_matsol_instance1(_threadargsproto_);
 /* connect range variables in _p that hoc is supposed to know about */
 static const char *_mechanism[] = {
 "7.7.0",
"GABAa",
 "Erev",
 "fflag",
 "sid",
 "cid",
 0,
 "i",
 "g",
 0,
 "Ron",
 "Roff",
 0,
 0};
 
extern Prop* need_memb(Symbol*);

static void nrn_alloc(Prop* _prop) {
	Prop *prop_ion;
	double *_p; Datum *_ppvar;
  if (nrn_point_prop_) {
	_prop->_alloc_seq = nrn_point_prop_->_alloc_seq;
	_p = nrn_point_prop_->param;
	_ppvar = nrn_point_prop_->dparam;
 }else{
 	_p = nrn_prop_data_alloc(_mechtype, 14, _prop);
 	/*initialize range parameters*/
 	Erev = -70;
 	fflag = 0;
 	sid = -1;
 	cid = -1;
  }
 	_prop->param = _p;
 	_prop->param_size = 14;
  if (!nrn_point_prop_) {
 	_ppvar = nrn_prop_datum_alloc(_mechtype, 4, _prop);
  }
 	_prop->dparam = _ppvar;
 	/*connect ionic variables to this model*/
 
}
 static void _initlists();
  /* some states have an absolute tolerance */
 static Symbol** _atollist;
 static HocStateTolerance _hoc_state_tol[] = {
 0,0
};
 
#define _tqitem &(_ppvar[2]._pvoid)
 static void _net_receive(Point_process*, double*, double);
 extern Symbol* hoc_lookup(const char*);
extern void _nrn_thread_reg(int, int, void(*)(Datum*));
extern void _nrn_thread_table_reg(int, void(*)(double*, Datum*, Datum*, NrnThread*, int));
extern void hoc_register_tolerance(int, HocStateTolerance*, Symbol***);
extern void _cvode_abstol( Symbol**, double*, int);

 void _GABAa_reg() {
	int _vectorized = 1;
  _initlists();
 	_pointtype = point_register_mech(_mechanism,
	 nrn_alloc,nrn_cur, nrn_jacob, nrn_state, nrn_init,
	 hoc_nrnpointerindex, 1,
	 _hoc_create_pnt, _hoc_destroy_pnt, _member_func);
 _mechtype = nrn_get_mechtype(_mechanism[1]);
     _nrn_setdata_reg(_mechtype, _setdata);
 #if NMODL_TEXT
  hoc_reg_nmodl_text(_mechtype, nmodl_file_text);
  hoc_reg_nmodl_filename(_mechtype, nmodl_filename);
#endif
  hoc_register_prop_size(_mechtype, 14, 4);
  hoc_register_dparam_semantics(_mechtype, 0, "area");
  hoc_register_dparam_semantics(_mechtype, 1, "pntproc");
  hoc_register_dparam_semantics(_mechtype, 2, "netsend");
  hoc_register_dparam_semantics(_mechtype, 3, "cvodeieq");
 	hoc_register_cvode(_mechtype, _ode_count, _ode_map, _ode_spec, _ode_matsol);
 	hoc_register_tolerance(_mechtype, _hoc_state_tol, &_atollist);
 pnt_receive[_mechtype] = _net_receive;
 pnt_receive_size[_mechtype] = 5;
 	hoc_register_var(hoc_scdoub, hoc_vdoub, hoc_intfunc);
 	ivoc_help("help ?1 GABAa /mnt/disk15tb/adam/git_workspace/netpyne/netpyne/tutorials/mod/GABAa.mod\n");
 hoc_register_limits(_mechtype, _hoc_parm_limits);
 hoc_register_units(_mechtype, _hoc_parm_units);
 }
static int _reset;
static char *modelname = "";

static int error;
static int _ninits = 0;
static int _match_recurse=1;
static void _modl_cleanup(){ _match_recurse=1;}
 
static int _ode_spec1(_threadargsproto_);
/*static int _ode_matsol1(_threadargsproto_);*/
 static int _slist1[2], _dlist1[2];
 static int release(_threadargsproto_);
 
/*CVODE*/
 static int _ode_spec1 (double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt) {int _reset = 0; {
   DRon = ( synon * Rinf - Ron ) / Rtau ;
   DRoff = - Beta * Roff ;
   }
 return _reset;
}
 static int _ode_matsol1 (double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt) {
 DRon = DRon  / (1. - dt*( ( ( ( - 1.0 ) ) ) / Rtau )) ;
 DRoff = DRoff  / (1. - dt*( ( - Beta )*( 1.0 ) )) ;
  return 0;
}
 /*END CVODE*/
 static int release (double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt) { {
    Ron = Ron + (1. - exp(dt*(( ( ( - 1.0 ) ) ) / Rtau)))*(- ( ( ( ( synon )*( Rinf ) ) ) / Rtau ) / ( ( ( ( - 1.0 ) ) ) / Rtau ) - Ron) ;
    Roff = Roff + (1. - exp(dt*(( - Beta )*( 1.0 ))))*(- ( 0.0 ) / ( ( - Beta )*( 1.0 ) ) - Roff) ;
   }
  return 0;
}
 
static void _net_receive (Point_process* _pnt, double* _args, double _lflag) 
{  double* _p; Datum* _ppvar; Datum* _thread; NrnThread* _nt;
   _thread = (Datum*)0; _nt = (NrnThread*)_pnt->_vnt;   _p = _pnt->_prop->param; _ppvar = _pnt->_prop->dparam;
  if (_tsav > t){ extern char* hoc_object_name(); hoc_execerror(hoc_object_name(_pnt->ob), ":Event arrived out of order. Must call ParallelContext.set_maxstep AFTER assigning minimum NetCon.delay");}
 _tsav = t;   if (_lflag == 1. ) {*(_tqitem) = 0;}
 {
   if ( t > 0.0 ) {
     if ( _lflag  == 0.0 ) {
       _args[2] = _args[2] + 1.0 ;
       if (  ! _args[1] ) {
         _args[3] = _args[3] * Exp1 ( _threadargscomma_ - Beta * ( t - _args[4] ) ) ;
         _args[4] = t ;
         _args[1] = 1.0 ;
         synon = synon + _args[0] ;
           if (nrn_netrec_state_adjust && !cvode_active_){
    /* discon state adjustment for cnexp case (rate uses no local variable) */
    double __state = Ron;
    double __primary = (Ron + _args[3]) - __state;
     __primary += ( 1. - exp( 0.5*dt*( ( ( ( - 1.0 ) ) ) / Rtau ) ) )*( - ( ( ( ( synon )*( Rinf ) ) ) / Rtau ) / ( ( ( ( - 1.0 ) ) ) / Rtau ) - __primary );
    Ron += __primary;
  } else {
 Ron = Ron + _args[3] ;
           }
   if (nrn_netrec_state_adjust && !cvode_active_){
    /* discon state adjustment for cnexp case (rate uses no local variable) */
    double __state = Roff;
    double __primary = (Roff - _args[3]) - __state;
     __primary += ( 1. - exp( 0.5*dt*( ( - Beta )*( 1.0 ) ) ) )*( - ( 0.0 ) / ( ( - Beta )*( 1.0 ) ) - __primary );
    Roff += __primary;
  } else {
 Roff = Roff - _args[3] ;
           }
 }
       net_send ( _tqitem, _args, _pnt, t +  Cdur , _args[2] ) ;
       }
     if ( _lflag  == _args[2] ) {
       _args[3] = _args[0] * Rinf + ( _args[3] - _args[0] * Rinf ) * Exp1 ( _threadargscomma_ - ( t - _args[4] ) / Rtau ) ;
       _args[4] = t ;
       synon = synon - _args[0] ;
         if (nrn_netrec_state_adjust && !cvode_active_){
    /* discon state adjustment for cnexp case (rate uses no local variable) */
    double __state = Ron;
    double __primary = (Ron - _args[3]) - __state;
     __primary += ( 1. - exp( 0.5*dt*( ( ( ( - 1.0 ) ) ) / Rtau ) ) )*( - ( ( ( ( synon )*( Rinf ) ) ) / Rtau ) / ( ( ( ( - 1.0 ) ) ) / Rtau ) - __primary );
    Ron += __primary;
  } else {
 Ron = Ron - _args[3] ;
         }
   if (nrn_netrec_state_adjust && !cvode_active_){
    /* discon state adjustment for cnexp case (rate uses no local variable) */
    double __state = Roff;
    double __primary = (Roff + _args[3]) - __state;
     __primary += ( 1. - exp( 0.5*dt*( ( - Beta )*( 1.0 ) ) ) )*( - ( 0.0 ) / ( ( - Beta )*( 1.0 ) ) - __primary );
    Roff += __primary;
  } else {
 Roff = Roff + _args[3] ;
         }
 _args[1] = 0.0 ;
       }
     }
   } }
 
double Exp1 ( _threadargsprotocomma_ double _lx ) {
   double _lExp1;
 if ( _lx < - 100.0 ) {
     _lExp1 = 0.0 ;
     }
   else if ( _lx > 100.0 ) {
     _lExp1 = exp ( 100.0 ) ;
     }
   else {
     _lExp1 = exp ( _lx ) ;
     }
   
return _lExp1;
 }
 
static double _hoc_Exp1(void* _vptr) {
 double _r;
   double* _p; Datum* _ppvar; Datum* _thread; NrnThread* _nt;
   _p = ((Point_process*)_vptr)->_prop->param;
  _ppvar = ((Point_process*)_vptr)->_prop->dparam;
  _thread = _extcall_thread;
  _nt = (NrnThread*)((Point_process*)_vptr)->_vnt;
 _r =  Exp1 ( _p, _ppvar, _thread, _nt, *getarg(1) );
 return(_r);
}
 
static int _ode_count(int _type){ return 2;}
 
static void _ode_spec(NrnThread* _nt, _Memb_list* _ml, int _type) {
   double* _p; Datum* _ppvar; Datum* _thread;
   Node* _nd; double _v; int _iml, _cntml;
  _cntml = _ml->_nodecount;
  _thread = _ml->_thread;
  for (_iml = 0; _iml < _cntml; ++_iml) {
    _p = _ml->_data[_iml]; _ppvar = _ml->_pdata[_iml];
    _nd = _ml->_nodelist[_iml];
    v = NODEV(_nd);
     _ode_spec1 (_p, _ppvar, _thread, _nt);
 }}
 
static void _ode_map(int _ieq, double** _pv, double** _pvdot, double* _pp, Datum* _ppd, double* _atol, int _type) { 
	double* _p; Datum* _ppvar;
 	int _i; _p = _pp; _ppvar = _ppd;
	_cvode_ieq = _ieq;
	for (_i=0; _i < 2; ++_i) {
		_pv[_i] = _pp + _slist1[_i];  _pvdot[_i] = _pp + _dlist1[_i];
		_cvode_abstol(_atollist, _atol, _i);
	}
 }
 
static void _ode_matsol_instance1(_threadargsproto_) {
 _ode_matsol1 (_p, _ppvar, _thread, _nt);
 }
 
static void _ode_matsol(NrnThread* _nt, _Memb_list* _ml, int _type) {
   double* _p; Datum* _ppvar; Datum* _thread;
   Node* _nd; double _v; int _iml, _cntml;
  _cntml = _ml->_nodecount;
  _thread = _ml->_thread;
  for (_iml = 0; _iml < _cntml; ++_iml) {
    _p = _ml->_data[_iml]; _ppvar = _ml->_pdata[_iml];
    _nd = _ml->_nodelist[_iml];
    v = NODEV(_nd);
 _ode_matsol_instance1(_threadargs_);
 }}

static void initmodel(double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt) {
  int _i; double _save;{
  Roff = Roff0;
  Ron = Ron0;
 {
   /* PROTECT */_NMODLMUTEXLOCK
 Rinf = Alpha / ( Alpha + Beta ) ;
   
 _NMODLMUTEXUNLOCK /* end PROTECT */
 /* PROTECT */_NMODLMUTEXLOCK
 Rtau = 1.0 / ( Alpha + Beta ) ;
   
 _NMODLMUTEXUNLOCK /* end PROTECT */
 synon = 0.0 ;
   }
 
}
}

static void nrn_init(NrnThread* _nt, _Memb_list* _ml, int _type){
double* _p; Datum* _ppvar; Datum* _thread;
Node *_nd; double _v; int* _ni; int _iml, _cntml;
#if CACHEVEC
    _ni = _ml->_nodeindices;
#endif
_cntml = _ml->_nodecount;
_thread = _ml->_thread;
for (_iml = 0; _iml < _cntml; ++_iml) {
 _p = _ml->_data[_iml]; _ppvar = _ml->_pdata[_iml];
 _tsav = -1e20;
#if CACHEVEC
  if (use_cachevec) {
    _v = VEC_V(_ni[_iml]);
  }else
#endif
  {
    _nd = _ml->_nodelist[_iml];
    _v = NODEV(_nd);
  }
 v = _v;
 initmodel(_p, _ppvar, _thread, _nt);
}
}

static double _nrn_current(double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt, double _v){double _current=0.;v=_v;{ {
   g = ( Ron + Roff ) * gmax ;
   i = g * ( v - Erev ) ;
   }
 _current += i;

} return _current;
}

static void nrn_cur(NrnThread* _nt, _Memb_list* _ml, int _type) {
double* _p; Datum* _ppvar; Datum* _thread;
Node *_nd; int* _ni; double _rhs, _v; int _iml, _cntml;
#if CACHEVEC
    _ni = _ml->_nodeindices;
#endif
_cntml = _ml->_nodecount;
_thread = _ml->_thread;
for (_iml = 0; _iml < _cntml; ++_iml) {
 _p = _ml->_data[_iml]; _ppvar = _ml->_pdata[_iml];
#if CACHEVEC
  if (use_cachevec) {
    _v = VEC_V(_ni[_iml]);
  }else
#endif
  {
    _nd = _ml->_nodelist[_iml];
    _v = NODEV(_nd);
  }
 _g = _nrn_current(_p, _ppvar, _thread, _nt, _v + .001);
 	{ _rhs = _nrn_current(_p, _ppvar, _thread, _nt, _v);
 	}
 _g = (_g - _rhs)/.001;
 _g *=  1.e2/(_nd_area);
 _rhs *= 1.e2/(_nd_area);
#if CACHEVEC
  if (use_cachevec) {
	VEC_RHS(_ni[_iml]) -= _rhs;
  }else
#endif
  {
	NODERHS(_nd) -= _rhs;
  }
 
}
 
}

static void nrn_jacob(NrnThread* _nt, _Memb_list* _ml, int _type) {
double* _p; Datum* _ppvar; Datum* _thread;
Node *_nd; int* _ni; int _iml, _cntml;
#if CACHEVEC
    _ni = _ml->_nodeindices;
#endif
_cntml = _ml->_nodecount;
_thread = _ml->_thread;
for (_iml = 0; _iml < _cntml; ++_iml) {
 _p = _ml->_data[_iml];
#if CACHEVEC
  if (use_cachevec) {
	VEC_D(_ni[_iml]) += _g;
  }else
#endif
  {
     _nd = _ml->_nodelist[_iml];
	NODED(_nd) += _g;
  }
 
}
 
}

static void nrn_state(NrnThread* _nt, _Memb_list* _ml, int _type) {
double* _p; Datum* _ppvar; Datum* _thread;
Node *_nd; double _v = 0.0; int* _ni; int _iml, _cntml;
#if CACHEVEC
    _ni = _ml->_nodeindices;
#endif
_cntml = _ml->_nodecount;
_thread = _ml->_thread;
for (_iml = 0; _iml < _cntml; ++_iml) {
 _p = _ml->_data[_iml]; _ppvar = _ml->_pdata[_iml];
 _nd = _ml->_nodelist[_iml];
#if CACHEVEC
  if (use_cachevec) {
    _v = VEC_V(_ni[_iml]);
  }else
#endif
  {
    _nd = _ml->_nodelist[_iml];
    _v = NODEV(_nd);
  }
 v=_v;
{
 {   release(_p, _ppvar, _thread, _nt);
  }}}

}

static void terminal(){}

static void _initlists(){
 double _x; double* _p = &_x;
 int _i; static int _first = 1;
  if (!_first) return;
 _slist1[0] = Ron_columnindex;  _dlist1[0] = DRon_columnindex;
 _slist1[1] = Roff_columnindex;  _dlist1[1] = DRoff_columnindex;
_first = 0;
}

#if defined(__cplusplus)
} /* extern "C" */
#endif

#if NMODL_TEXT
static const char* nmodl_filename = "/mnt/disk15tb/adam/git_workspace/netpyne/netpyne/tutorials/mod/GABAa.mod";
static const char* nmodl_file_text = 
  "NEURON {  POINT_PROCESS GABAa }\n"
  "\n"
  "PARAMETER {\n"
  "  Cdur	= 1.08	(ms)		: transmitter duration (rising phase)\n"
  "  Alpha	= 1.	(/ms mM)	: forward (binding) rate\n"
  "  Beta	= 0.5	(/ms)		: backward (unbinding) rate\n"
  "  Erev	= -70	(mV)		: reversal potential\n"
  "}\n"
  "\n"
  ":::INCLUDE \"netcon.inc\"\n"
  ":::realpath /mnt/disk15tb/adam/git_workspace/netpyne/netpyne/tutorials/mod/netcon.inc\n"
  ": $Id: netcon.inc,v 1.16 2010/03/28 16:19:27 billl Exp $\n"
  "\n"
  "COMMENT\n"
  "USAGE: for most receptors\n"
  " *****************************************************************************\n"
  "    NEURON {\n"
  "      POINT_PROCESS NAME\n"
  "    }\n"
  "\n"
  "    PARAMETER {\n"
  "      Cdur	= 1.08	(ms)		: transmitter duration (rising phase)\n"
  "      Alpha	= 1	(/ms mM)	: forward (binding) rate\n"
  "      Beta	= 0.02	(/ms)		: backward (unbinding) rate\n"
  "      Erev	= -80	(mV)		: reversal potential\n"
  "    }\n"
  "    \n"
  "    INCLUDE \"netcon.inc\"\n"
  " *****************************************************************************\n"
  "\n"
  "USAGE: for NMDA receptor\n"
  " *****************************************************************************\n"
  "    NEURON{ POINT_PROCESS NMDA\n"
  "      RANGE B \n"
  "    }\n"
  "\n"
  "    PARAMETER {\n"
  "      mg        = 1.    (mM)     : external magnesium concentration\n"
  "      Cdur	= 1.	(ms)	 : transmitter duration (rising phase)\n"
  "      Alpha	= 4.	(/ms mM) : forward (binding) rate\n"
  "      Beta	= 0.0067 (/ms)	 : backward (unbinding) rate 1/150\n"
  "      Erev	= 0.	(mV)	 : reversal potential\n"
  "    }\n"
  "\n"
  "    ASSIGNED { B }\n"
  "\n"
  "    INCLUDE \"netcon.inc\"\n"
  "    : EXTRA BREAKPOINT MUST BE BELOW THE INCLUDE\n"
  "    BREAKPOINT {\n"
  "      rates(v)\n"
  "      g = g * B : but don't really need to readjust conductance\n"
  "      i = i * B : i = g*(v - Erev)\n"
  "    }\n"
  "\n"
  "    PROCEDURE rates(v(mV)) {\n"
  "      TABLE B\n"
  "      DEPEND mg\n"
  "      FROM -100 TO 80 WITH 180\n"
  "      B = 1 / (1 + Exp1(0.062 (/mV) * -v) * (mg / 3.57 (mM)))\n"
  "    }\n"
  " *****************************************************************************\n"
  "ENDCOMMENT\n"
  "\n"
  "INDEPENDENT {t FROM 0 TO 1 WITH 1 (ms)}\n"
  "\n"
  "NEURON {\n"
  "  RANGE g, Erev, fflag\n"
  "  RANGE sid,cid\n"
  "  NONSPECIFIC_CURRENT i\n"
  "  GLOBAL Cdur, Alpha, Beta, Rinf, Rtau, gmax\n"
  "}\n"
  "\n"
  "UNITS {\n"
  "  (nA) = (nanoamp)\n"
  "  (mV) = (millivolt)\n"
  "  (umho) = (micromho)\n"
  "  (mM) = (milli/liter)\n"
  "}\n"
  "\n"
  "PARAMETER {\n"
  "  fflag = 0\n"
  "  sid = -1 (1) : synapse id, from cell template\n"
  "  cid = -1 (1) : id of cell to which this synapse is attached\n"
  "  gmax = 1 (S)\n"
  "}\n"
  "\n"
  "ASSIGNED {\n"
  "  v		(mV)		: postsynaptic voltage\n"
  "  i 		(nA)		: current = g*(v - Erev)\n"
  "  g 		(umho)		: conductance\n"
  "  Rinf				: steady state channels open\n"
  "  Rtau		(ms)		: time constant of channel binding\n"
  "  synon\n"
  "}\n"
  "\n"
  "STATE {Ron Roff}\n"
  "\n"
  "INITIAL {\n"
  "  PROTECT Rinf = Alpha / (Alpha + Beta)\n"
  "  PROTECT Rtau = 1 / (Alpha + Beta)\n"
  "  synon = 0\n"
  "}\n"
  "\n"
  "BREAKPOINT {\n"
  "  SOLVE release METHOD cnexp\n"
  "  g = (Ron + Roff)*gmax\n"
  "  i = g*(v - Erev)\n"
  "}\n"
  "\n"
  "DERIVATIVE release {\n"
  "  Ron' = (synon*Rinf - Ron)/Rtau\n"
  "  Roff' = -Beta*Roff\n"
  "}\n"
  "\n"
  ": following supports both saturation from single input and\n"
  ": summation from multiple inputs\n"
  ": if spike occurs during CDur then new off time is t + CDur\n"
  ": ie. transmitter concatenates but does not summate\n"
  ": Note: automatic initialization of all reference args to 0 except first\n"
  "\n"
  "NET_RECEIVE (weight, on, nspike, r0, t0 (ms)) {\n"
  "  : flag is an implicit argument of NET_RECEIVE and  normally 0\n"
  "  if (t>0) { : bug fix so that init doesn't send a false event\n"
  "    if (flag == 0) { : a spike, so turn on if not already in a Cdur pulse\n"
  "      nspike = nspike + 1\n"
  "      if (!on) {\n"
  "        r0 = r0*Exp1(-Beta*(t - t0))\n"
  "        t0 = t\n"
  "        on = 1\n"
  "        synon = synon + weight\n"
  "        Ron = Ron + r0\n"
  "        Roff = Roff - r0\n"
  "      }\n"
  "      : come again in Cdur with flag = current value of nspike\n"
  "      net_send(Cdur, nspike)\n"
  "    }\n"
  "    if (flag == nspike) { : if this associated with last spike then turn off\n"
  "      r0 = weight*Rinf + (r0 - weight*Rinf)*Exp1(-(t - t0)/Rtau)\n"
  "      t0 = t\n"
  "      synon = synon - weight\n"
  "      Ron = Ron - r0\n"
  "      Roff = Roff + r0\n"
  "      on = 0\n"
  "    }\n"
  "  }\n"
  "}\n"
  "\n"
  "FUNCTION Exp1(x) {   \n"
  "  if (x < -100) {\n"
  "    Exp1  = 0\n"
  "  } else if (x > 100) {\n"
  "    Exp1 = exp(100)\n"
  "  } else{\n"
  "    Exp1 = exp(x)\n"
  "  }\n"
  "}\n"
  ":::end INCLUDE netcon.inc\n"
  " \n"
  ":** GABAb\n"
  "\n"
  "\n"
  "\n"
  "\n"
  "\n"
  "\n"
  ;
#endif
