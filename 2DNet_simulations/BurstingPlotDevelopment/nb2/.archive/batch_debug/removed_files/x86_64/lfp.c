/* Created by Language version: 7.7.0 */
/* NOT VECTORIZED */
#define NRN_VECTORIZED 0
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
 
#define nrn_init _nrn_init__lfp
#define _nrn_initial _nrn_initial__lfp
#define nrn_cur _nrn_cur__lfp
#define _nrn_current _nrn_current__lfp
#define nrn_jacob _nrn_jacob__lfp
#define nrn_state _nrn_state__lfp
#define _net_receive _net_receive__lfp 
 
#define _threadargscomma_ /**/
#define _threadargsprotocomma_ /**/
#define _threadargs_ /**/
#define _threadargsproto_ /**/
 	/*SUPPRESS 761*/
	/*SUPPRESS 762*/
	/*SUPPRESS 763*/
	/*SUPPRESS 765*/
	 extern double *getarg();
 static double *_p; static Datum *_ppvar;
 
#define t nrn_threads->_t
#define dt nrn_threads->_dt
#define initial_part_line _p[0]
#define initial_part_line_columnindex 0
#define initial_part_rc _p[1]
#define initial_part_rc_columnindex 1
#define lfp_line _p[2]
#define lfp_line_columnindex 2
#define lfp_point _p[3]
#define lfp_point_columnindex 3
#define lfp_rc _p[4]
#define lfp_rc_columnindex 4
#define initial_part_point _p[5]
#define initial_part_point_columnindex 5
#define _g _p[6]
#define _g_columnindex 6
#define transmembrane_current	*_ppvar[0]._pval
#define _p_transmembrane_current	_ppvar[0]._pval
 
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
 static int hoc_nrnpointerindex =  0;
 /* external NEURON variables */
 /* declaration of user functions */
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

 extern void _nrn_setdata_reg(int, void(*)(Prop*));
 static void _setdata(Prop* _prop) {
 _p = _prop->param; _ppvar = _prop->dparam;
 }
 static void _hoc_setdata() {
 Prop *_prop, *hoc_getdata_range(int);
 _prop = hoc_getdata_range(_mechtype);
   _setdata(_prop);
 hoc_retpushx(1.);
}
 /* connect user functions to hoc names */
 static VoidFunc hoc_intfunc[] = {
 "setdata_lfp", _hoc_setdata,
 0, 0
};
 /* declare global and static user variables */
 /* some parameters have upper and lower limits */
 static HocParmLimits _hoc_parm_limits[] = {
 0,0,0
};
 static HocParmUnits _hoc_parm_units[] = {
 0,0
};
 static double v = 0;
 /* connect global user variables to hoc */
 static DoubScal hoc_scdoub[] = {
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
 /* connect range variables in _p that hoc is supposed to know about */
 static const char *_mechanism[] = {
 "7.7.0",
"lfp",
 0,
 "initial_part_line_lfp",
 "initial_part_rc_lfp",
 "lfp_line_lfp",
 "lfp_point_lfp",
 "lfp_rc_lfp",
 "initial_part_point_lfp",
 0,
 0,
 "transmembrane_current_lfp",
 0};
 
extern Prop* need_memb(Symbol*);

static void nrn_alloc(Prop* _prop) {
	Prop *prop_ion;
	double *_p; Datum *_ppvar;
 	_p = nrn_prop_data_alloc(_mechtype, 7, _prop);
 	/*initialize range parameters*/
 	_prop->param = _p;
 	_prop->param_size = 7;
 	_ppvar = nrn_prop_datum_alloc(_mechtype, 1, _prop);
 	_prop->dparam = _ppvar;
 	/*connect ionic variables to this model*/
 
}
 static void _initlists();
 extern Symbol* hoc_lookup(const char*);
extern void _nrn_thread_reg(int, int, void(*)(Datum*));
extern void _nrn_thread_table_reg(int, void(*)(double*, Datum*, Datum*, NrnThread*, int));
extern void hoc_register_tolerance(int, HocStateTolerance*, Symbol***);
extern void _cvode_abstol( Symbol**, double*, int);

 void _lfp_reg() {
	int _vectorized = 0;
  _initlists();
 	register_mech(_mechanism, nrn_alloc,nrn_cur, nrn_jacob, nrn_state, nrn_init, hoc_nrnpointerindex, 0);
 _mechtype = nrn_get_mechtype(_mechanism[1]);
     _nrn_setdata_reg(_mechtype, _setdata);
 #if NMODL_TEXT
  hoc_reg_nmodl_text(_mechtype, nmodl_file_text);
  hoc_reg_nmodl_filename(_mechtype, nmodl_filename);
#endif
  hoc_register_prop_size(_mechtype, 7, 1);
  hoc_register_dparam_semantics(_mechtype, 0, "pointer");
 	hoc_register_var(hoc_scdoub, hoc_vdoub, hoc_intfunc);
 	ivoc_help("help ?1 lfp /mnt/disk15tb/adam/git_workspace/netpyne/netpyne/tutorials/mod/lfp.mod\n");
 hoc_register_limits(_mechtype, _hoc_parm_limits);
 hoc_register_units(_mechtype, _hoc_parm_units);
 }
static int _reset;
static char *modelname = "";

static int error;
static int _ninits = 0;
static int _match_recurse=1;
static void _modl_cleanup(){ _match_recurse=1;}

static void initmodel() {
  int _i; double _save;_ninits++;
{

}
}

static void nrn_init(NrnThread* _nt, _Memb_list* _ml, int _type){
Node *_nd; double _v; int* _ni; int _iml, _cntml;
#if CACHEVEC
    _ni = _ml->_nodeindices;
#endif
_cntml = _ml->_nodecount;
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
 v = _v;
 initmodel();
}}

static double _nrn_current(double _v){double _current=0.;v=_v;{
} return _current;
}

static void nrn_cur(NrnThread* _nt, _Memb_list* _ml, int _type){
Node *_nd; int* _ni; double _rhs, _v; int _iml, _cntml;
#if CACHEVEC
    _ni = _ml->_nodeindices;
#endif
_cntml = _ml->_nodecount;
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
 
}}

static void nrn_jacob(NrnThread* _nt, _Memb_list* _ml, int _type){
Node *_nd; int* _ni; int _iml, _cntml;
#if CACHEVEC
    _ni = _ml->_nodeindices;
#endif
_cntml = _ml->_nodecount;
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
 
}}

static void nrn_state(NrnThread* _nt, _Memb_list* _ml, int _type){
Node *_nd; double _v = 0.0; int* _ni; int _iml, _cntml;
#if CACHEVEC
    _ni = _ml->_nodeindices;
#endif
_cntml = _ml->_nodecount;
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
 {
   lfp_point = transmembrane_current * initial_part_point * 1e-1 ;
   lfp_line = transmembrane_current * initial_part_line * 1e-1 ;
   lfp_rc = transmembrane_current * initial_part_rc * 1e-3 ;
   }
}}

}

static void terminal(){}

static void _initlists() {
 int _i; static int _first = 1;
  if (!_first) return;
_first = 0;
}

#if NMODL_TEXT
static const char* nmodl_filename = "/mnt/disk15tb/adam/git_workspace/netpyne/netpyne/tutorials/mod/lfp.mod";
static const char* nmodl_file_text = 
  ": lfp.mod\n"
  "\n"
  "COMMENT\n"
  "LFPsim - Simulation scripts to compute Local Field Potentials (LFP) from cable compartmental models of neurons and networks implemented in NEURON simulation environment.\n"
  "\n"
  "LFPsim works reliably on biophysically detailed multi-compartmental neurons with ion channels in some or all compartments.\n"
  "\n"
  "Last updated 12-March-2016\n"
  "Developed by : Harilal Parasuram & Shyam Diwakar\n"
  "Computational Neuroscience & Neurophysiology Lab, School of Biotechnology, Amrita University, India.\n"
  "Email: harilalp@am.amrita.edu; shyam@amrita.edu\n"
  "www.amrita.edu/compneuro \n"
  "ENDCOMMENT\n"
  "\n"
  "NEURON {\n"
  "	SUFFIX lfp\n"
  "	POINTER transmembrane_current\n"
  "	RANGE lfp_line,lfp_point,lfp_rc,initial_part_point, initial_part_line, initial_part_rc\n"
  "	\n"
  "}\n"
  "\n"
  "\n"
  "ASSIGNED {\n"
  "\n"
  "	initial_part_line \n"
  "	initial_part_rc\n"
  "	transmembrane_current \n"
  "	lfp_line\n"
  "	lfp_point\n"
  "	lfp_rc\n"
  "	initial_part_point\n"
  "\n"
  "\n"
  "}\n"
  "\n"
  "BREAKPOINT { \n"
  "\n"
  "	:Point Source Approximation 	\n"
  "	lfp_point =   transmembrane_current * initial_part_point * 1e-1   : So the calculated signal will be in nV\n"
  "\n"
  "	:Line Source Approximation\n"
  "	lfp_line =   transmembrane_current * initial_part_line  * 1e-1  : So the calculated signal will be in nV\n"
  "\n"
  "	:RC\n"
  "	lfp_rc =   transmembrane_current * initial_part_rc * 1e-3 : So the calculated signal will be in nV\n"
  "\n"
  "}\n"
  "\n"
  "\n"
  ;
#endif
