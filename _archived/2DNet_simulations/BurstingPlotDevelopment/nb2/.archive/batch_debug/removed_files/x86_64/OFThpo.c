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
 
#define nrn_init _nrn_init__OFPO
#define _nrn_initial _nrn_initial__OFPO
#define nrn_cur _nrn_cur__OFPO
#define _nrn_current _nrn_current__OFPO
#define nrn_jacob _nrn_jacob__OFPO
#define nrn_state _nrn_state__OFPO
#define _net_receive _net_receive__OFPO 
#define iassign iassign__OFPO 
#define states states__OFPO 
#define version version__OFPO 
 
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
#define gkbase _p[0]
#define gkbase_columnindex 0
#define taugka _p[1]
#define taugka_columnindex 1
#define spkht _p[2]
#define spkht_columnindex 2
#define tauk _p[3]
#define tauk_columnindex 3
#define ek _p[4]
#define ek_columnindex 4
#define vth _p[5]
#define vth_columnindex 5
#define refrac _p[6]
#define refrac_columnindex 6
#define inrefrac _p[7]
#define inrefrac_columnindex 7
#define apdur _p[8]
#define apdur_columnindex 8
#define gkinc _p[9]
#define gkinc_columnindex 9
#define tauvtha _p[10]
#define tauvtha_columnindex 10
#define vthinc _p[11]
#define vthinc_columnindex 11
#define ik _p[12]
#define ik_columnindex 12
#define i _p[13]
#define i_columnindex 13
#define gkmin _p[14]
#define gkmin_columnindex 14
#define gnamax _p[15]
#define gnamax_columnindex 15
#define ena _p[16]
#define ena_columnindex 16
#define gk _p[17]
#define gk_columnindex 17
#define vthadapt _p[18]
#define vthadapt_columnindex 18
#define gkadapt _p[19]
#define gkadapt_columnindex 19
#define iother _p[20]
#define iother_columnindex 20
#define Dgk _p[21]
#define Dgk_columnindex 21
#define Dvthadapt _p[22]
#define Dvthadapt_columnindex 22
#define Dgkadapt _p[23]
#define Dgkadapt_columnindex 23
#define v _p[24]
#define v_columnindex 24
#define _g _p[25]
#define _g_columnindex 25
#define _tsav _p[26]
#define _tsav_columnindex 26
#define _nd_area  *_ppvar[0]._pval
#define _ion_iother	*_ppvar[2]._pval
#define _ion_diotherdv	*_ppvar[3]._pval
 
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
 static double _hoc_fflag(void*);
 static double _hoc_iassign(void*);
 static double _hoc_version(void*);
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
 "fflag", _hoc_fflag,
 "iassign", _hoc_iassign,
 "version", _hoc_version,
 0, 0
};
#define fflag fflag_OFPO
 extern double fflag( _threadargsproto_ );
 /* declare global and static user variables */
#define checkref checkref_OFPO
 double checkref = 1;
#define kadapt kadapt_OFPO
 double kadapt = 0.007;
#define verbose verbose_OFPO
 double verbose = 0;
 /* some parameters have upper and lower limits */
 static HocParmLimits _hoc_parm_limits[] = {
 0,0,0
};
 static HocParmUnits _hoc_parm_units[] = {
 "kadapt_OFPO", "uS",
 "gkbase", "uS",
 "taugka", "ms",
 "spkht", "mV",
 "tauk", "ms",
 "ek", "mV",
 "vth", "mV",
 "refrac", "ms",
 "apdur", "ms",
 "gkinc", "uS",
 "tauvtha", "ms",
 "ik", "nA",
 "i", "nA",
 "gkmin", "uS",
 0,0
};
 static double delta_t = 0.01;
 static double gkadapt0 = 0;
 static double gk0 = 0;
 static double vthadapt0 = 0;
 /* connect global user variables to hoc */
 static DoubScal hoc_scdoub[] = {
 "kadapt_OFPO", &kadapt_OFPO,
 "verbose_OFPO", &verbose_OFPO,
 "checkref_OFPO", &checkref_OFPO,
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
 
#define _watch_array _ppvar + 5 
 static void _watch_alloc(Datum*);
 extern void hoc_reg_watch_allocate(int, void(*)(Datum*)); static void _hoc_destroy_pnt(void* _vptr) {
   Prop* _prop = ((Point_process*)_vptr)->_prop;
   if (_prop) { _nrn_free_watch(_prop->dparam, 5, 2);}
   destroy_point_process(_vptr);
}
 
static int _ode_count(int);
static void _ode_map(int, double**, double**, double*, Datum*, double*, int);
static void _ode_spec(NrnThread*, _Memb_list*, int);
static void _ode_matsol(NrnThread*, _Memb_list*, int);
 
#define _cvode_ieq _ppvar[7]._i
 static void _ode_matsol_instance1(_threadargsproto_);
 /* connect range variables in _p that hoc is supposed to know about */
 static const char *_mechanism[] = {
 "7.7.0",
"OFPO",
 "gkbase",
 "taugka",
 "spkht",
 "tauk",
 "ek",
 "vth",
 "refrac",
 "inrefrac",
 "apdur",
 "gkinc",
 "tauvtha",
 "vthinc",
 "ik",
 "i",
 "gkmin",
 "gnamax",
 "ena",
 0,
 0,
 "gk",
 "vthadapt",
 "gkadapt",
 0,
 0};
 static Symbol* _other_sym;
 
extern Prop* need_memb(Symbol*);

static void nrn_alloc(Prop* _prop) {
	Prop *prop_ion;
	double *_p; Datum *_ppvar;
  if (nrn_point_prop_) {
	_prop->_alloc_seq = nrn_point_prop_->_alloc_seq;
	_p = nrn_point_prop_->param;
	_ppvar = nrn_point_prop_->dparam;
 }else{
 	_p = nrn_prop_data_alloc(_mechtype, 27, _prop);
 	/*initialize range parameters*/
 	gkbase = 0.06;
 	taugka = 100;
 	spkht = 55;
 	tauk = 2.3;
 	ek = -70;
 	vth = -40;
 	refrac = 2.7;
 	inrefrac = 0;
 	apdur = 0.9;
 	gkinc = 0.006;
 	tauvtha = 1;
 	vthinc = 0;
 	ik = 0;
 	i = 0;
 	gkmin = 1e-05;
 	gnamax = 0;
 	ena = 0;
  }
 	_prop->param = _p;
 	_prop->param_size = 27;
  if (!nrn_point_prop_) {
 	_ppvar = nrn_prop_datum_alloc(_mechtype, 8, _prop);
  }
 	_prop->dparam = _ppvar;
 	/*connect ionic variables to this model*/
 prop_ion = need_memb(_other_sym);
 	_ppvar[2]._pval = &prop_ion->param[3]; /* iother */
 	_ppvar[3]._pval = &prop_ion->param[4]; /* _ion_diotherdv */
 
}
 static void _initlists();
  /* some states have an absolute tolerance */
 static Symbol** _atollist;
 static HocStateTolerance _hoc_state_tol[] = {
 0,0
};
 
#define _tqitem &(_ppvar[4]._pvoid)
 static void _net_receive(Point_process*, double*, double);
 static void _update_ion_pointer(Datum*);
 extern Symbol* hoc_lookup(const char*);
extern void _nrn_thread_reg(int, int, void(*)(Datum*));
extern void _nrn_thread_table_reg(int, void(*)(double*, Datum*, Datum*, NrnThread*, int));
extern void hoc_register_tolerance(int, HocStateTolerance*, Symbol***);
extern void _cvode_abstol( Symbol**, double*, int);

 void _OFThpo_reg() {
	int _vectorized = 1;
  _initlists();
 	ion_reg("other", 1.0);
 	_other_sym = hoc_lookup("other_ion");
 	_pointtype = point_register_mech(_mechanism,
	 nrn_alloc,nrn_cur, nrn_jacob, nrn_state, nrn_init,
	 hoc_nrnpointerindex, 1,
	 _hoc_create_pnt, _hoc_destroy_pnt, _member_func);
 _mechtype = nrn_get_mechtype(_mechanism[1]);
     _nrn_setdata_reg(_mechtype, _setdata);
     _nrn_thread_reg(_mechtype, 2, _update_ion_pointer);
 #if NMODL_TEXT
  hoc_reg_nmodl_text(_mechtype, nmodl_file_text);
  hoc_reg_nmodl_filename(_mechtype, nmodl_filename);
#endif
  hoc_register_prop_size(_mechtype, 27, 8);
  hoc_reg_watch_allocate(_mechtype, _watch_alloc);
  hoc_register_dparam_semantics(_mechtype, 0, "area");
  hoc_register_dparam_semantics(_mechtype, 1, "pntproc");
  hoc_register_dparam_semantics(_mechtype, 2, "other_ion");
  hoc_register_dparam_semantics(_mechtype, 3, "other_ion");
  hoc_register_dparam_semantics(_mechtype, 4, "netsend");
  hoc_register_dparam_semantics(_mechtype, 5, "watch");
  hoc_register_dparam_semantics(_mechtype, 6, "watch");
  hoc_register_dparam_semantics(_mechtype, 7, "cvodeieq");
 	hoc_register_cvode(_mechtype, _ode_count, _ode_map, _ode_spec, _ode_matsol);
 	hoc_register_tolerance(_mechtype, _hoc_state_tol, &_atollist);
 add_nrn_has_net_event(_mechtype);
 pnt_receive[_mechtype] = _net_receive;
 pnt_receive_size[_mechtype] = 1;
 	hoc_register_var(hoc_scdoub, hoc_vdoub, hoc_intfunc);
 	ivoc_help("help ?1 OFPO /mnt/disk15tb/adam/git_workspace/netpyne/netpyne/tutorials/mod/OFThpo.mod\n");
 hoc_register_limits(_mechtype, _hoc_parm_limits);
 hoc_register_units(_mechtype, _hoc_parm_units);
 }
static int _reset;
static char *modelname = "OF Threshold Spiking";

static int error;
static int _ninits = 0;
static int _match_recurse=1;
static void _modl_cleanup(){ _match_recurse=1;}
static int iassign(_threadargsproto_);
static int version(_threadargsproto_);
 
static int _ode_spec1(_threadargsproto_);
/*static int _ode_matsol1(_threadargsproto_);*/
 static int _slist1[3], _dlist1[3];
 static int states(_threadargsproto_);
 
/*CVODE*/
 static int _ode_spec1 (double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt) {int _reset = 0; {
   Dgk = - gk / tauk ;
   Dgkadapt = ( gkbase - gkadapt ) / taugka ;
   Dvthadapt = ( vth - vthadapt ) / tauvtha ;
   }
 return _reset;
}
 static int _ode_matsol1 (double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt) {
 Dgk = Dgk  / (1. - dt*( ( - 1.0 ) / tauk )) ;
 Dgkadapt = Dgkadapt  / (1. - dt*( ( ( ( - 1.0 ) ) ) / taugka )) ;
 Dvthadapt = Dvthadapt  / (1. - dt*( ( ( ( - 1.0 ) ) ) / tauvtha )) ;
  return 0;
}
 /*END CVODE*/
 static int states (double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt) { {
    gk = gk + (1. - exp(dt*(( - 1.0 ) / tauk)))*(- ( 0.0 ) / ( ( - 1.0 ) / tauk ) - gk) ;
    gkadapt = gkadapt + (1. - exp(dt*(( ( ( - 1.0 ) ) ) / taugka)))*(- ( ( ( gkbase ) ) / taugka ) / ( ( ( ( - 1.0 ) ) ) / taugka ) - gkadapt) ;
    vthadapt = vthadapt + (1. - exp(dt*(( ( ( - 1.0 ) ) ) / tauvtha)))*(- ( ( ( vth ) ) / tauvtha ) / ( ( ( ( - 1.0 ) ) ) / tauvtha ) - vthadapt) ;
   }
  return 0;
}
 
static int  iassign ( _threadargsproto_ ) {
   ik = gk * ( v - ek ) ;
   i = ik ;
   iother = i ;
    return 0; }
 
static double _hoc_iassign(void* _vptr) {
 double _r;
   double* _p; Datum* _ppvar; Datum* _thread; NrnThread* _nt;
   _p = ((Point_process*)_vptr)->_prop->param;
  _ppvar = ((Point_process*)_vptr)->_prop->dparam;
  _thread = _extcall_thread;
  _nt = (NrnThread*)((Point_process*)_vptr)->_vnt;
 _r = 1.;
 iassign ( _p, _ppvar, _thread, _nt );
 return(_r);
}
 
static double _watch1_cond(Point_process* _pnt) {
 	double* _p; Datum* _ppvar; Datum* _thread; NrnThread* _nt;
	_thread= (Datum*)0; _nt = (NrnThread*)_pnt->_vnt;
 	_p = _pnt->_prop->param; _ppvar = _pnt->_prop->dparam;
	v = NODEV(_pnt->node);
	return  ( v ) - ( vthadapt ) ;
}
 
static void _net_receive (Point_process* _pnt, double* _args, double _lflag) 
{  double* _p; Datum* _ppvar; Datum* _thread; NrnThread* _nt;
   int _watch_rm = 0;
   _thread = (Datum*)0; _nt = (NrnThread*)_pnt->_vnt;   _p = _pnt->_prop->param; _ppvar = _pnt->_prop->dparam;
  if (_tsav > t){ extern char* hoc_object_name(); hoc_execerror(hoc_object_name(_pnt->ob), ":Event arrived out of order. Must call ParallelContext.set_maxstep AFTER assigning minimum NetCon.delay");}
 _tsav = t;  v = NODEV(_pnt->node);
   if (_lflag == 1. ) {*(_tqitem) = 0;}
 {
   if ( _lflag  == 1.0 ) {
       _nrn_watch_activate(_watch_array, _watch1_cond, 1, _pnt, _watch_rm++, 2.0);
 }
   else if ( _lflag  == 2.0 ) {
     if ( inrefrac  == 0.0 ) {
       net_event ( _pnt, t ) ;
       net_send ( _tqitem, _args, _pnt, t +  apdur , 3.0 ) ;
       net_send ( _tqitem, _args, _pnt, t +  refrac , 4.0 ) ;
       inrefrac = 1.0 ;
       if ( verbose ) {
         printf ( "spike at t=%g\n" , t ) ;
         }
       }
     else {
       if ( verbose ) {
         printf ( "in refrac @ t = %g, no spike\n" , t ) ;
         }
       }
     }
   else if ( _lflag  == 3.0 ) {
       if (nrn_netrec_state_adjust && !cvode_active_){
    /* discon state adjustment for cnexp case (rate uses no local variable) */
    double __state = gkadapt;
    double __primary = (gkadapt + gkinc) - __state;
     __primary += ( 1. - exp( 0.5*dt*( ( ( ( - 1.0 ) ) ) / taugka ) ) )*( - ( ( ( gkbase ) ) / taugka ) / ( ( ( ( - 1.0 ) ) ) / taugka ) - __primary );
    gkadapt += __primary;
  } else {
 gkadapt = gkadapt + gkinc ;
       }
   if (nrn_netrec_state_adjust && !cvode_active_){
    /* discon state adjustment for cnexp case (rate uses no local variable) */
    double __state = vthadapt;
    double __primary = (vthadapt + vthinc) - __state;
     __primary += ( 1. - exp( 0.5*dt*( ( ( ( - 1.0 ) ) ) / tauvtha ) ) )*( - ( ( ( vth ) ) / tauvtha ) / ( ( ( ( - 1.0 ) ) ) / tauvtha ) - __primary );
    vthadapt += __primary;
  } else {
 vthadapt = vthadapt + vthinc ;
       }
   if (nrn_netrec_state_adjust && !cvode_active_){
    /* discon state adjustment for cnexp case (rate uses no local variable) */
    double __state = gk;
    double __primary = (gkadapt) - __state;
     __primary += ( 1. - exp( 0.5*dt*( ( - 1.0 ) / tauk ) ) )*( - ( 0.0 ) / ( ( - 1.0 ) / tauk ) - __primary );
    gk += __primary;
  } else {
 gk = gkadapt ;
       }
 if ( verbose ) {
       printf ( "end of action potential @ t = %g\n" , t ) ;
       }
     }
   else if ( _lflag  == 4.0 ) {
     inrefrac = 0.0 ;
     if ( verbose ) {
       printf ( "refrac over @ t = %g\n" , t ) ;
       }
     if ( checkref  && v > vthadapt ) {
       net_send ( _tqitem, _args, _pnt, t +  0.0 , 2.0 ) ;
       }
     }
   } 
 NODEV(_pnt->node) = v;
 }
 
double fflag ( _threadargsproto_ ) {
   double _lfflag;
 _lfflag = 1.0 ;
   
return _lfflag;
 }
 
static double _hoc_fflag(void* _vptr) {
 double _r;
   double* _p; Datum* _ppvar; Datum* _thread; NrnThread* _nt;
   _p = ((Point_process*)_vptr)->_prop->param;
  _ppvar = ((Point_process*)_vptr)->_prop->dparam;
  _thread = _extcall_thread;
  _nt = (NrnThread*)((Point_process*)_vptr)->_vnt;
 _r =  fflag ( _p, _ppvar, _thread, _nt );
 return(_r);
}
 
static int  version ( _threadargsproto_ ) {
   printf ( "$Id: OFThpo.mod,v 1.3 2009/03/13 11:57:56 billl Exp $ " ) ;
    return 0; }
 
static double _hoc_version(void* _vptr) {
 double _r;
   double* _p; Datum* _ppvar; Datum* _thread; NrnThread* _nt;
   _p = ((Point_process*)_vptr)->_prop->param;
  _ppvar = ((Point_process*)_vptr)->_prop->dparam;
  _thread = _extcall_thread;
  _nt = (NrnThread*)((Point_process*)_vptr)->_vnt;
 _r = 1.;
 version ( _p, _ppvar, _thread, _nt );
 return(_r);
}
 
static void _watch_alloc(Datum* _ppvar) {
  Point_process* _pnt = (Point_process*)_ppvar[1]._pvoid;
   _nrn_watch_allocate(_watch_array, _watch1_cond, 1, _pnt, 2.0);
 }

 
static int _ode_count(int _type){ return 3;}
 
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
	for (_i=0; _i < 3; ++_i) {
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
 extern void nrn_update_ion_pointer(Symbol*, Datum*, int, int);
 static void _update_ion_pointer(Datum* _ppvar) {
   nrn_update_ion_pointer(_other_sym, _ppvar, 2, 3);
   nrn_update_ion_pointer(_other_sym, _ppvar, 3, 4);
 }

static void initmodel(double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt) {
  int _i; double _save;{
  gkadapt = gkadapt0;
  gk = gk0;
  vthadapt = vthadapt0;
 {
   net_send ( _tqitem, (double*)0, _ppvar[1]._pvoid, t +  0.0 , 1.0 ) ;
   gk = 0.0 ;
   gkadapt = gkbase ;
   vthadapt = vth ;
   ik = 0.0 ;
   i = 0.0 ;
   iother = 0.0 ;
   inrefrac = 0.0 ;
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
   if ( gk < gkmin ) {
     gk = gkmin ;
     }
   if ( gkadapt < gkbase ) {
     gkadapt = gkbase ;
     }
   if ( vthadapt < vth ) {
     vthadapt = vth ;
     }
   iassign ( _threadargs_ ) ;
   }
 _current += iother;

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
 	{ double _diother;
  _diother = iother;
 _rhs = _nrn_current(_p, _ppvar, _thread, _nt, _v);
  _ion_diotherdv += (_diother - iother)/.001 * 1.e2/ (_nd_area);
 	}
 _g = (_g - _rhs)/.001;
  _ion_iother += iother * 1.e2/ (_nd_area);
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
 {   states(_p, _ppvar, _thread, _nt);
  } }}

}

static void terminal(){}

static void _initlists(){
 double _x; double* _p = &_x;
 int _i; static int _first = 1;
  if (!_first) return;
 _slist1[0] = gk_columnindex;  _dlist1[0] = Dgk_columnindex;
 _slist1[1] = gkadapt_columnindex;  _dlist1[1] = Dgkadapt_columnindex;
 _slist1[2] = vthadapt_columnindex;  _dlist1[2] = Dvthadapt_columnindex;
_first = 0;
}

#if defined(__cplusplus)
} /* extern "C" */
#endif

#if NMODL_TEXT
static const char* nmodl_filename = "/mnt/disk15tb/adam/git_workspace/netpyne/netpyne/tutorials/mod/OFThpo.mod";
static const char* nmodl_file_text = 
  ": $Id: OFThpo.mod,v 1.3 2009/03/13 11:57:56 billl Exp $ \n"
  "\n"
  "COMMENT\n"
  "based on Otto Friesen Neurodynamix model\n"
  "spiking portion of cell model\n"
  "variant on OFThresh.mod that uses paste-on instead of calculated spike\n"
  "ENDCOMMENT\n"
  "\n"
  "TITLE OF Threshold Spiking\n"
  "\n"
  "UNITS {\n"
  "    (mV) = (millivolt)\n"
  "    (nA) = (nanoamp)\n"
  "    (uS) = (microsiemens)\n"
  "}\n"
  "\n"
  "NEURON {\n"
  "  POINT_PROCESS OFPO\n"
  "  USEION other WRITE iother VALENCE 1.0\n"
  "\n"
  "  RANGE gkbase             : base level of k+ conductance after a spike\n"
  "  RANGE gkmin              : min level of k+ conductance can decay to after a spike\n"
  "  RANGE vth                : threshold for spike\n"
  "  RANGE tauvtha, vthinc    : used for threshold adaptation\n"
  "  RANGE taugka, gkinc      : used for gk adaptation (gkadapt state var)\n"
  "  RANGE ik,ek              : k-related variables\n"
  "  RANGE tauk               : tau for k current\n"
  "  RANGE i,spkht            : current, spike height\n"
  "  RANGE refrac             : duration of absolute refractory period\n"
  "  RANGE inrefrac           : if in refractory period\n"
  "  RANGE apdur              : action potential duration\n"
  "  RANGE ena,gnamax         : na-related variables -- NOT used\n"
  "\n"
  "  GLOBAL verbose\n"
  "  GLOBAL checkref          : check for spikes @ end of refrac\n"
  "}\n"
  "\n"
  "ASSIGNED { \n"
  "  v (mV)\n"
  "  iother (nA)\n"
  "}\n"
  "\n"
  "STATE { gk vthadapt gkadapt }\n"
  "\n"
  "PARAMETER {\n"
  "  gkbase=0.060(uS) : Max Potassium conductance\n"
  "  taugka=100 (ms) : Time constant of adaptation\n"
  "  kadapt=0.007(uS) : Amount of adaptation for potassium\n"
  "  spkht = 55(mV)\n"
  "  tauk=2.3 (ms) : Time constant for potassium current \n"
  "  ek = -70(mV)\n"
  "  vth = -40(mV)\n"
  "  refrac = 2.7(ms)\n"
  "  inrefrac = 0\n"
  "  verbose = 0\n"
  "  apdur = 0.9 (ms)\n"
  "  gkinc = 0.006(uS)\n"
  "  tauvtha = 1(ms)\n"
  "  vthinc = 0\n"
  "  ik = 0(nA)\n"
  "  i = 0(nA)\n"
  "  gkmin = 0.00001(uS)\n"
  "  checkref = 1\n"
  "  gnamax = 0 : Na not used -- back compatibility\n"
  "  ena = 0\n"
  "}\n"
  "\n"
  "BREAKPOINT {\n"
  "  SOLVE states METHOD cnexp\n"
  "  if( gk < gkmin ) { gk = gkmin }\n"
  "  if( gkadapt < gkbase ) { gkadapt = gkbase }\n"
  "  if( vthadapt < vth ) { vthadapt = vth }\n"
  "  iassign()\n"
  "}\n"
  "\n"
  "INITIAL {\n"
  "  net_send(0,1)\n"
  "  gk = 0(uS)\n"
  "  gkadapt = gkbase\n"
  "  vthadapt = vth\n"
  "  ik = 0  \n"
  "  i = 0\n"
  "  iother = 0\n"
  "  inrefrac = 0\n"
  "}\n"
  "\n"
  "DERIVATIVE states {  \n"
  "  gk' = -gk/tauk\n"
  "  gkadapt' = (gkbase - gkadapt)/taugka\n"
  "  vthadapt' = (vth - vthadapt)/tauvtha\n"
  "}\n"
  "\n"
  "PROCEDURE iassign () {\n"
  "  ik = gk*(v-ek)\n"
  "  i = ik\n"
  "  iother = i\n"
  "}\n"
  "\n"
  "NET_RECEIVE (w) {\n"
  "  if (flag == 1) {\n"
  "    WATCH (v > vthadapt) 2\n"
  "  } else if (flag == 2) {  :v > threshold\n"
  "    if(inrefrac == 0) {      :if not in refractory period, spike\n"
  "      net_event(t)           :send spike event\n"
  "      net_send(apdur,3)      :send event for end of action potential\n"
  "      net_send(refrac,4)     :send event for end of refractory period\n"
  "      inrefrac=1             :in refractory period\n"
  "      if( verbose ) { printf(\"spike at t=%g\\n\",t) }\n"
  "    } else {\n"
  "      if( verbose ) { printf(\"in refrac @ t = %g, no spike\\n\",t) }\n"
  "    }\n"
  "  } else if(flag == 3) {   :end of action potential\n"
  "    gkadapt = gkadapt + gkinc\n"
  "    vthadapt = vthadapt + vthinc : threshold adaptation\n"
  "    gk = gkadapt           :turn gk to max after action potential over\n"
  "    if (verbose) { printf(\"end of action potential @ t = %g\\n\",t) }\n"
  "  } else if(flag == 4) {   :end of refractory period\n"
  "    inrefrac = 0           :set inrefrac flag off\n"
  "    if( verbose ) { printf(\"refrac over @ t = %g\\n\",t) }\n"
  "    :check for new spike @ end of refrac\n"
  "    if(checkref && v > vthadapt) { net_send(0,2) }\n"
  "  }\n"
  "}\n"
  "\n"
  "FUNCTION fflag () { fflag=1 }\n"
  "\n"
  "PROCEDURE version () {\n"
  "  printf(\"$Id: OFThpo.mod,v 1.3 2009/03/13 11:57:56 billl Exp $ \")\n"
  "}\n"
  "\n"
  "\n"
  "\n"
  "\n"
  "\n"
  "\n"
  ;
#endif
