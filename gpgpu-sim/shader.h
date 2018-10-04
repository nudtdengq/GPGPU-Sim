// Copyright (c) 2009-2011, Tor M. Aamodt, Wilson W.L. Fung, Andrew Turner,
// Ali Bakhoda 
// The University of British Columbia
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// Redistributions of source code must retain the above copyright notice, this
// list of conditions and the following disclaimer.
// Redistributions in binary form must reproduce the above copyright notice, this
// list of conditions and the following disclaimer in the documentation and/or
// other materials provided with the distribution.
// Neither the name of The University of British Columbia nor the names of its
// contributors may be used to endorse or promote products derived from this
// software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
// ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
// WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#ifndef SHADER_H
#define SHADER_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <map>
#include <set>
#include <vector>
#include <list>
#include <bitset>
#include <utility>
#include <algorithm>
#include <deque>
#include <iostream>

//#include "../cuda-sim/ptx.tab.h"

#include "delayqueue.h"
#include "stack.h"
#include "dram.h"
#include "../abstract_hardware_model.h"
#include "scoreboard.h"
#include "mem_fetch.h"
#include "stats.h"
#include "gpu-cache.h"
#include "traffic_breakdown.h"



#define NO_OP_FLAG            0xFF

/* READ_PACKET_SIZE:
   bytes: 6 address (flit can specify chanel so this gives up to ~2GB/channel, so good for now),
          2 bytes   [shaderid + mshrid](14 bits) + req_size(0-2 bits if req_size variable) - so up to 2^14 = 16384 mshr total 
 */

#define READ_PACKET_SIZE 8

//WRITE_PACKET_SIZE: bytes: 6 address, 2 miscelaneous. 
#define WRITE_PACKET_SIZE 8

#define WRITE_MASK_SIZE 8


class thread_ctx_t {
public:
   unsigned m_cta_id; // hardware CTA this thread belongs

   // per thread stats (ac stands for accumulative).
   unsigned n_insn;
   unsigned n_insn_ac;
   unsigned n_l1_mis_ac;
   unsigned n_l1_mrghit_ac;
   unsigned n_l1_access_ac; 

   bool m_active; 
};

class shd_warp_t {
public:
    shd_warp_t( class shader_core_ctx *shader, unsigned warp_size);
    void reset()
    {
        assert( m_stores_outstanding==0);
        assert( m_inst_in_pipeline==0);
        m_imiss_pending=false;
        m_warp_id=(unsigned)-1;
        m_dynamic_warp_id = (unsigned)-1;
        n_completed = m_warp_size; 
        m_n_atomic=0;
        m_membar=false;
        m_done_exit=true;
        m_last_fetch=0;
        m_next=0;
        m_inst_at_barrier=NULL;
    }
    void init( address_type start_pc,
               unsigned cta_id,
               unsigned wid,
               const std::bitset<MAX_WARP_SIZE> &active,
               unsigned dynamic_warp_id )
    {
        m_cta_id=cta_id;
        m_warp_id=wid;
        m_dynamic_warp_id=dynamic_warp_id;
        m_next_pc=start_pc;
        assert( n_completed >= active.count() );
        assert( n_completed <= m_warp_size);
        n_completed   -= active.count(); // active threads are not yet completed
        m_active_threads = active;
        m_done_exit=false;
        ////////DQ_code//////warp////
        //printf("start____warp:%u;CTA_id:%u\n",wid,cta_id);
        ///////DQ_code/////warp//////
    }

    bool functional_done() const;
    bool waiting(); // not const due to membar
    bool hardware_done() const;

    bool done_exit() const { return m_done_exit; }
    void set_done_exit() { m_done_exit=true; }

    void print( FILE *fout ) const;
    void print_ibuffer( FILE *fout ) const;

    unsigned get_n_completed() const { return n_completed; }
    void set_completed( unsigned lane ) 
    { 
        assert( m_active_threads.test(lane) );
        m_active_threads.reset(lane);
        n_completed++; 
    }

    void set_last_fetch( unsigned long long sim_cycle ) { m_last_fetch=sim_cycle; }

    unsigned get_n_atomic() const { return m_n_atomic; }
    void inc_n_atomic() { m_n_atomic++; }
    void dec_n_atomic(unsigned n) { m_n_atomic-=n; }

    void set_membar() { m_membar=true; }
    void clear_membar() { m_membar=false; }
    bool get_membar() const { return m_membar; }
    address_type get_pc() const { return m_next_pc; }
    void set_next_pc( address_type pc ) { m_next_pc = pc; }

    void store_info_of_last_inst_at_barrier(const warp_inst_t *pI){ m_inst_at_barrier = pI;}
    const warp_inst_t * restore_info_of_last_inst_at_barrier(){ return m_inst_at_barrier;}

    void ibuffer_fill( unsigned slot, const warp_inst_t *pI )
    {
       assert(slot < IBUFFER_SIZE );
       m_ibuffer[slot].m_inst=pI;
       m_ibuffer[slot].m_valid=true;
       m_next=0; 
    }
    bool ibuffer_empty() const
    {
        for( unsigned i=0; i < IBUFFER_SIZE; i++) 
            if(m_ibuffer[i].m_valid) 
                return false;
        return true;
    }
    void ibuffer_flush()
    {
        for(unsigned i=0;i<IBUFFER_SIZE;i++) {
            if( m_ibuffer[i].m_valid )
                dec_inst_in_pipeline();
            m_ibuffer[i].m_inst=NULL; 
            m_ibuffer[i].m_valid=false; 
        }
    }
    const warp_inst_t *ibuffer_next_inst() { return m_ibuffer[m_next].m_inst; }
    bool ibuffer_next_valid() { return m_ibuffer[m_next].m_valid; }
    void ibuffer_free()
    {
        m_ibuffer[m_next].m_inst = NULL;
        m_ibuffer[m_next].m_valid = false;
    }
    void ibuffer_step() { m_next = (m_next+1)%IBUFFER_SIZE; }

    bool imiss_pending() const { return m_imiss_pending; }
    void set_imiss_pending() { m_imiss_pending=true; }
    void clear_imiss_pending() { m_imiss_pending=false; }

    bool stores_done() const { return m_stores_outstanding == 0; }
    void inc_store_req() { m_stores_outstanding++; }
    void dec_store_req() 
    {
        assert( m_stores_outstanding > 0 );
        m_stores_outstanding--;
    }

    unsigned num_inst_in_buffer() const
    {
    	unsigned count=0;
        for(unsigned i=0;i<IBUFFER_SIZE;i++) {
            if( m_ibuffer[i].m_valid )
            	count++;
        }
    	return count;
    }
    unsigned num_inst_in_pipeline() const { return m_inst_in_pipeline;}
    unsigned num_issued_inst_in_pipeline() const {return (num_inst_in_pipeline()-num_inst_in_buffer());}
    bool inst_in_pipeline() const { return m_inst_in_pipeline > 0; }
    void inc_inst_in_pipeline() { m_inst_in_pipeline++; }
    void dec_inst_in_pipeline() 
    {
        assert( m_inst_in_pipeline > 0 );
        m_inst_in_pipeline--;
    }

    unsigned get_cta_id() const { return m_cta_id; }

    unsigned get_dynamic_warp_id() const { return m_dynamic_warp_id; }
    unsigned get_warp_id() const { return m_warp_id; }

private:
    static const unsigned IBUFFER_SIZE=2;
    class shader_core_ctx *m_shader;
    unsigned m_cta_id;
    unsigned m_warp_id;
    unsigned m_warp_size;
    unsigned m_dynamic_warp_id;

    address_type m_next_pc;
    unsigned n_completed;          // number of threads in warp completed
    std::bitset<MAX_WARP_SIZE> m_active_threads;

    bool m_imiss_pending;
    
    struct ibuffer_entry {
       ibuffer_entry() { m_valid = false; m_inst = NULL; }
       const warp_inst_t *m_inst;
       bool m_valid;
    };

    const warp_inst_t *m_inst_at_barrier;
    ibuffer_entry m_ibuffer[IBUFFER_SIZE]; 
    unsigned m_next;
                                   
    unsigned m_n_atomic;           // number of outstanding atomic operations 
    bool     m_membar;             // if true, warp is waiting at memory barrier

    bool m_done_exit; // true once thread exit has been registered for threads in this warp

    unsigned long long m_last_fetch;

    unsigned m_stores_outstanding; // number of store requests sent but not yet acknowledged
    unsigned m_inst_in_pipeline;
};



inline unsigned hw_tid_from_wid(unsigned wid, unsigned warp_size, unsigned i){return wid * warp_size + i;};
inline unsigned wid_from_hw_tid(unsigned tid, unsigned warp_size){return tid/warp_size;};

const unsigned WARP_PER_CTA_MAX = 48;
typedef std::bitset<WARP_PER_CTA_MAX> warp_set_t;

int register_bank(int regnum, int wid, unsigned num_banks, unsigned bank_warp_shift);

class shader_core_ctx;
class shader_core_config;
class shader_core_stats;

enum scheduler_prioritization_type
{
    SCHEDULER_PRIORITIZATION_LRR = 0, // Loose Round Robin
    SCHEDULER_PRIORITIZATION_SRR, // Strict Round Robin
    SCHEDULER_PRIORITIZATION_GTO, // Greedy Then Oldest
    SCHEDULER_PRIORITIZATION_GTLRR, // Greedy Then Loose Round Robin
    SCHEDULER_PRIORITIZATION_GTY, // Greedy Then Youngest
    SCHEDULER_PRIORITIZATION_OLDEST, // Oldest First
    SCHEDULER_PRIORITIZATION_YOUNGEST, // Youngest First
};

// Each of these corresponds to a string value in the gpgpsim.config file
// For example - to specify the LRR scheudler the config must contain lrr
enum concrete_scheduler
{
    CONCRETE_SCHEDULER_LRR = 0,
    CONCRETE_SCHEDULER_GTO,
    CONCRETE_SCHEDULER_TWO_LEVEL_ACTIVE,
    CONCRETE_SCHEDULER_WARP_LIMITING,
    NUM_CONCRETE_SCHEDULERS
};

class scheduler_unit { //this can be copied freely, so can be used in std containers.
public:
    scheduler_unit(shader_core_stats* stats, shader_core_ctx* shader, 
                   Scoreboard* scoreboard, simt_stack** simt, 
                   std::vector<shd_warp_t>* warp, 
                   register_set* sp_out,
                   register_set* sfu_out,
                   register_set* mem_out,
                   int id) 
        : m_supervised_warps(), m_stats(stats), m_shader(shader),
        m_scoreboard(scoreboard), m_simt_stack(simt), /*m_pipeline_reg(pipe_regs),*/ m_warp(warp),
        m_sp_out(sp_out),m_sfu_out(sfu_out),m_mem_out(mem_out), m_id(id){
    	//DQ_config//
    	sub_config=200;
        valid=new unsigned*[sub_config];//[200] [4];
        count=new unsigned*[sub_config];
        flag=new int*[sub_config];// [200] [4];///DQ_code////DQ_config///
        row_flag=new int*[sub_config];//[200] [4];
        bank_flag=new int*[sub_config];

        for(int i=0;i<sub_config;i++){
        	valid[i]=new unsigned[16];
        	count[i]=new unsigned[16];
        	flag[i]=new int[16];
        	row_flag[i]=new int[16];
        	bank_flag[i]=new int[16];
        }

    }
    virtual ~scheduler_unit(){}
    virtual void add_supervised_warp_id(int i) {
        m_supervised_warps.push_back(&warp(i));
    }
    virtual void done_adding_supervised_warps() {
        m_last_supervised_issued = m_supervised_warps.end();
    }


    // The core scheduler cycle method is meant to be common between
    // all the derived schedulers.  The scheduler's behaviour can be
    // modified by changing the contents of the m_next_cycle_prioritized_warps list.
    void cycle();

    // These are some common ordering fucntions that the
    // higher order schedulers can take advantage of
    template < typename T >
    void order_lrr( typename std::vector< T >& result_list,
                    const typename std::vector< T >& input_list,
                    const typename std::vector< T >::const_iterator& last_issued_from_input,
                    unsigned num_warps_to_add );
    
    enum OrderingType 
    {
        // The item that issued last is prioritized first then the sorted result
        // of the priority_function
        ORDERING_GREEDY_THEN_PRIORITY_FUNC = 0,
        // No greedy scheduling based on last to issue. Only the priority function determines
        // priority
        ORDERED_PRIORITY_FUNC_ONLY,
        NUM_ORDERING,
    };
    template < typename U >
    void order_by_priority( std::vector< U >& result_list,
                            const typename std::vector< U >& input_list,
                            const typename std::vector< U >::const_iterator& last_issued_from_input,
                            unsigned num_warps_to_add,
                            OrderingType age_ordering,
                            bool (*priority_func)(U lhs, U rhs) );
    static bool sort_warps_by_oldest_dynamic_id(shd_warp_t* lhs, shd_warp_t* rhs);

    // Derived classes can override this function to populate
    // m_supervised_warps with their scheduling policies
    virtual void order_warps() = 0;

protected:
    virtual void do_on_warp_issued( unsigned warp_id,
                                    unsigned num_issued,
                                    const std::vector< shd_warp_t* >::const_iterator& prioritized_iter );
    inline int get_sid() const;
protected:
    shd_warp_t& warp(int i);

    // This is the prioritized warp list that is looped over each cycle to determine
    // which warp gets to issue.
    std::vector< shd_warp_t* > m_next_cycle_prioritized_warps;
    // The m_supervised_warps list is all the warps this scheduler is supposed to
    // arbitrate between.  This is useful in systems where there is more than
    // one warp scheduler. In a single scheduler system, this is simply all
    // the warps assigned to this core.
    std::vector< shd_warp_t* > m_supervised_warps;
    // This is the iterator pointer to the last supervised warp you issued
    std::vector< shd_warp_t* >::const_iterator m_last_supervised_issued;
    shader_core_stats *m_stats;
    shader_core_ctx* m_shader;
    // these things should become accessors: but would need a bigger rearchitect of how shader_core_ctx interacts with its parts.
    Scoreboard* m_scoreboard; 
    simt_stack** m_simt_stack;
    //warp_inst_t** m_pipeline_reg;
    std::vector<shd_warp_t>* m_warp;
    register_set* m_sp_out;
    register_set* m_sfu_out;
    register_set* m_mem_out;

    int m_id;

private:

    ////DQ_code///flag///
    unsigned** valid;//[200] [4];
    unsigned** count;//[200] [4];
    int sub_config;
    int inc;
    unsigned row_nu;
    unsigned bank;
    int** flag;// [200] [4];///DQ_code////DQ_config///
    int**  row_flag;//[200] [4];
    int**  bank_flag;//[200] [4];
    int flagCount;
};

class lrr_scheduler : public scheduler_unit {
public:
	lrr_scheduler ( shader_core_stats* stats, shader_core_ctx* shader,
                    Scoreboard* scoreboard, simt_stack** simt,
                    std::vector<shd_warp_t>* warp,
                    register_set* sp_out,
                    register_set* sfu_out,
                    register_set* mem_out,
                    int id )
	: scheduler_unit ( stats, shader, scoreboard, simt, warp, sp_out, sfu_out, mem_out, id ){}
	virtual ~lrr_scheduler () {}
	virtual void order_warps ();
    virtual void done_adding_supervised_warps() {
        m_last_supervised_issued = m_supervised_warps.end();
    }
};

class gto_scheduler : public scheduler_unit {
public:
	gto_scheduler ( shader_core_stats* stats, shader_core_ctx* shader,
                    Scoreboard* scoreboard, simt_stack** simt,
                    std::vector<shd_warp_t>* warp,
                    register_set* sp_out,
                    register_set* sfu_out,
                    register_set* mem_out,
                    int id )
	: scheduler_unit ( stats, shader, scoreboard, simt, warp, sp_out, sfu_out, mem_out, id ){}
	virtual ~gto_scheduler () {}
	virtual void order_warps ();
    virtual void done_adding_supervised_warps() {
        m_last_supervised_issued = m_supervised_warps.begin();
    }

};


class two_level_active_scheduler : public scheduler_unit {
public:
	two_level_active_scheduler ( shader_core_stats* stats, shader_core_ctx* shader,
                          Scoreboard* scoreboard, simt_stack** simt,
                          std::vector<shd_warp_t>* warp,
                          register_set* sp_out,
                          register_set* sfu_out,
                          register_set* mem_out,
                          int id,
                          char* config_str )
	: scheduler_unit ( stats, shader, scoreboard, simt, warp, sp_out, sfu_out, mem_out, id ),
	  m_pending_warps() 
    {
        unsigned inner_level_readin;
        unsigned outer_level_readin; 
        int ret = sscanf( config_str,
                          "two_level_active:%d:%d:%d",
                          &m_max_active_warps,
                          &inner_level_readin,
                          &outer_level_readin);
        assert( 3 == ret );
        m_inner_level_prioritization=(scheduler_prioritization_type)inner_level_readin;
        m_outer_level_prioritization=(scheduler_prioritization_type)outer_level_readin;
    }
	virtual ~two_level_active_scheduler () {}
    virtual void order_warps();
	void add_supervised_warp_id(int i) {
        if ( m_next_cycle_prioritized_warps.size() < m_max_active_warps ) {
            m_next_cycle_prioritized_warps.push_back( &warp(i) );
        } else {
		    m_pending_warps.push_back(&warp(i));
        }
	}
    virtual void done_adding_supervised_warps() {
        m_last_supervised_issued = m_supervised_warps.begin();
    }

protected:
    virtual void do_on_warp_issued( unsigned warp_id,
                                    unsigned num_issued,
                                    const std::vector< shd_warp_t* >::const_iterator& prioritized_iter );

private:
	std::deque< shd_warp_t* > m_pending_warps;
    scheduler_prioritization_type m_inner_level_prioritization;
    scheduler_prioritization_type m_outer_level_prioritization;
	unsigned m_max_active_warps;
};

// Static Warp Limiting Scheduler
class swl_scheduler : public scheduler_unit {
public:
	swl_scheduler ( shader_core_stats* stats, shader_core_ctx* shader,
                    Scoreboard* scoreboard, simt_stack** simt,
                    std::vector<shd_warp_t>* warp,
                    register_set* sp_out,
                    register_set* sfu_out,
                    register_set* mem_out,
                    int id,
                    char* config_string );
	virtual ~swl_scheduler () {}
	virtual void order_warps ();
    virtual void done_adding_supervised_warps() {
        m_last_supervised_issued = m_supervised_warps.begin();
    }

protected:
    scheduler_prioritization_type m_prioritization;
    unsigned m_num_warps_to_limit;
};


class opndcoll_rfu_t { // operand collector based register file unit
public:
	//DQ_code


   // constructors
   opndcoll_rfu_t()
   {
      m_num_banks=0;
      m_shader=NULL;
      m_initialized=false;
//DQ_code//config
      sub_config=200;
      Nbank=20;

   }

   /*unsigned return_subbank(){
	   return m_shader.subbank_size;
   }*/

////DQ_code/////check_bank_idle////

   bool check_bank_idle( const warp_inst_t &inst );
   bool check_latency(const warp_inst_t &inst, unsigned latency );
   void reset_check();

   void add_cu_set(unsigned cu_set, unsigned num_cu, unsigned num_dispatch);
   typedef std::vector<register_set*> port_vector_t;
   typedef std::vector<unsigned int> uint_vector_t;
   void add_port( port_vector_t & input, port_vector_t & ouput, uint_vector_t cu_sets);
   void init( unsigned num_banks, shader_core_ctx *shader );

   // modifiers
   bool writeback( const warp_inst_t &warp ); // might cause stall 

   void step()
   {
        dispatch_ready_cu();
        allocate_reads();

        for( unsigned p = 0 ; p < m_in_ports.size(); p++ ) 
            allocate_cu( p );
        process_banks();
   }

   void dump( FILE *fp ) const
   {
      fprintf(fp,"\n");
      fprintf(fp,"Operand Collector State:\n");
      for( unsigned n=0; n < m_cu.size(); n++ ) {
         fprintf(fp,"   CU-%2u: ", n);
         m_cu[n]->dump(fp,m_shader);
      }
      m_arbiter.dump(fp);
   }

   shader_core_ctx *shader_core() { return m_shader; }

private:

   void process_banks()
   {
	   /*for (i=0; i<=m_num_banks; ++i){}
	    * if (m_allocated_bank[i].is_write())
	    *int count = 0;
	    *count += 1;
	    *count
	    *
	    */
      m_arbiter.reset_alloction();
   }

   unsigned subbank_nu(int regnum, int wid, unsigned num_banks, unsigned bank_warp_shift);

//   unsigned subbank_nu(int regnum, int wid, unsigned num_banks, unsigned bank_warp_shift){
//
//	    unsigned subbank_size = sub_size;
//	    if(m_shader->interleave==1){
//	    	unsigned row_nu = (regnum%64+wid%16)/num_banks*12+wid%4*48+wid/4;//DQ_code_interleave_warped_register_mapping
//	    }else{
//	    	unsigned row_nu = (regnum%64+wid%16)/num_banks+wid*4;//DQ_code sequence register mapping
//	    }
////	    unsigned row_nu = (regnum+wid%16)/num_banks*12+wid%4*48+wid/4;//DQ_code_interleave_warped_register_mapping
//	    //unsigned row_nu = (regnum+wid)/num_banks+wid*4;//DQ_code sequence register mapping
//	    		//+wid/4*4+(wid/4*48+wid%4*4);//16 into 4//DQ_code_interleave_warped_register_mapping
//	    //unsigned bank = (regnum+wid-1)%num_banks;
//	    //DQ_code//1 buffer+N nvm
//	    unsigned sub_nu=row_nu/subbank_size;
//	    //int bank = regnum;
//	    //if (bank_warp_shift)
//	         //bank += wid;
//	    return sub_nu;
//
//   }

   void dispatch_ready_cu();
   void allocate_cu( unsigned port );
   void allocate_reads();

   // types

   class collector_unit_t;

   class op_t {
   public:

      op_t() { m_valid = false; }
      op_t( collector_unit_t *cu, unsigned op, unsigned reg, unsigned num_banks, unsigned bank_warp_shift )
      {
    	  /////DQ///////with cu////////
         m_valid = true;
         m_warp=NULL;
         m_cu = cu;
         m_operand = op;
         m_register = reg;
         m_bank = register_bank(reg,cu->get_warp_id(),num_banks,bank_warp_shift);
      }
      op_t( const warp_inst_t *warp, unsigned reg, unsigned num_banks, unsigned bank_warp_shift )
      {
    	  /////DQ//////without cu/////
    	//  std::cout<<"warp_id"<<warp->warp_id()<<std::endl;
    	  //std::cout<<"reg:"<<reg<<"num_banks:"<<num_banks<<"bank_warp_shift:"<<bank_warp_shift<<std::endl;

         m_valid=true;
         m_warp=warp;
         m_register=reg;
         m_cu=NULL;
         m_operand = -1;
         m_bank = register_bank(reg,warp->warp_id(),num_banks,bank_warp_shift);
      //   printf("______________________op_t\n");
      }

      // accessors
      bool valid() const { return m_valid; }
      unsigned get_reg() const
      {
         assert( m_valid );
         return m_register;
      }
      unsigned get_wid() const
      {

          if( m_warp ) return m_warp->warp_id();

          else if( m_cu ) return m_cu->get_warp_id();
          else abort();
      }
      unsigned get_active_count() const
      {
          if( m_warp ) return m_warp->active_count();
          else if( m_cu ) return m_cu->get_active_count();
          else abort();
      }
      const active_mask_t & get_active_mask()
      {
          if( m_warp ) return m_warp->get_active_mask();
          else if( m_cu ) return m_cu->get_active_mask();
          else abort();
      }
      unsigned get_sp_op() const
      {
          if( m_warp ) return m_warp->sp_op;
          else if( m_cu ) return m_cu->get_sp_op();
          else abort();
      }
      unsigned get_oc_id() const { return m_cu->get_id(); }
      unsigned get_bank() const { return m_bank; }
      unsigned get_operand() const { return m_operand; }
      void dump(FILE *fp) const 
      {
         if(m_cu) 
            fprintf(fp," <R%u, CU:%u, w:%02u> ", m_register,m_cu->get_id(),m_cu->get_warp_id());
         else if( !m_warp->empty() )
            fprintf(fp," <R%u, wid:%02u> ", m_register,m_warp->warp_id() );
      }
      std::string get_reg_string() const
      {
         char buffer[64];
         snprintf(buffer,64,"R%u", m_register);
         return std::string(buffer);
      }

      // modifiers
      void reset() { m_valid = false; }
   private:
      bool m_valid;
      collector_unit_t  *m_cu; 
      const warp_inst_t *m_warp;
      unsigned  m_operand; // operand offset in instruction. e.g., add r1,r2,r3; r2 is oprd 0, r3 is 1 (r1 is dst)
      unsigned  m_register;
      unsigned  m_bank;

   };

   enum alloc_t {
      NO_ALLOC,
      READ_ALLOC,
      WRITE_ALLOC,
   };

   class allocation_t {
   public:

      allocation_t() {
    	  m_allocation = NO_ALLOC;

      	  sub_config=200;
      	  sub_m_allocation = new enum alloc_t[sub_config];
      	  for(int i=0; i<sub_config;i++){
      		  sub_m_allocation[i] = NO_ALLOC;
      	  }
      }
   /*   /////DQ_Code/////////
  	   NV_Nu={1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31};
  	   SRAM_Nu={2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32};
      for (int i = 0; i< 32; ++i)
            	   {

            	   	flag[i][0]= 0;/////1
            	   	flag[i][1] = 1;

            	   	flag[i][2]= 0;/////2
            	   	flag[i][3] = 1;

            	   	flag[i][4]= 0;/////3
            	   	flag[i][5] = 1;

            	   	flag[i][6]= 0;/////4
            	   	flag[i][7] = 1;

            	   	flag[i][8]= 0;/////5
            	   	flag[i][9] = 1;

            	   	flag[i][10]= 0;/////6
            	   	flag[i][11] = 1;

            	   	flag[i][12]= 0;/////7
            	   	flag[i][13] = 1;

            	   	flag[i][14]= 0;/////8
            	   	flag[i][15] = 1;

            	   	flag[i][16]= 0;/////9
            	   	flag[i][17] = 1;

            	   	flag[i][18]= 0;/////10
            	   	flag[i][19] = 1;

            	   	flag[i][20]= 0;/////11
            	   	flag[i][21] = 1;

            	   	flag[i][22]= 0;/////12
            	   	flag[i][23] = 1;

            	   	flag[i][24]= 0;/////13
            	   	flag[i][25] = 1;

            	   	flag[i][26]= 0;/////14
            	   	flag[i][27] = 1;

            	   	flag[i][28]= 0;/////15
            	   	flag[i][29] = 1;

            	   	flag[i][30]= 0;/////16
            	   	flag[i][31] = 1;
            	   }

*/
            	   /////DQ_Code/////////
      
      bool is_read() const { return m_allocation==READ_ALLOC; }
      bool is_write() const {return m_allocation==WRITE_ALLOC; }
      bool is_free() const {return m_allocation==NO_ALLOC; }
      //DQ_code//
      bool sub_is_read(unsigned subbank) const { return sub_m_allocation[subbank]==READ_ALLOC; }
      bool sub_is_write(unsigned subbank) const {return sub_m_allocation[subbank]==WRITE_ALLOC; }
      bool sub_is_free(unsigned subbank) const {return sub_m_allocation[subbank]==NO_ALLOC;}


/*     ///////////DQ_code/////is_flag////////
      bool is_flag()
      {


    	  for(int i = 0; i<16; ++i)
    	  {
    		 if (reg_out == NV_Nu[i]  && flag[warp_ID_out][reg_out] == 1)
    		 {
    			 Num_NV = NV_Nu[i];
    			 Num_Sram = SRAM_Nu[i];e_reg;
    Scoreboard               *m_scoreboa
    			 flag[warp_ID_out][Num_NV] = ~flag [warp_ID_out][Num_NV];
    			 flag[warp_ID_out][Num_Sram] = ~flag [warp_ID_out][Num_Sram];
    		 }
    		 if (reg_out == SRAM_Nu[i]  &&  flag[warp_ID_out][reg_out] == 1)
    		 {
    			 Num_NV = NV_Nu[i];
    			 Num_Sram = SRAM_Nu[i];
    			 flag[warp_ID_out][Num_NV] = ~flag [warp_ID_out][Num_NV];
    			 flag[warp_ID_out][Num_Sram] = ~flag [warp_ID_out][Num_Sram];
    		 }
    		 else{
    			 printf("______DQ_____allocation->is_flag______");
    		 }
    	  }
    	  return NV==1;
      }
      ///////////DQ_code/////is_flag////////
*/




      void dump(FILE *fp) const {
         if( m_allocation == NO_ALLOC ) { fprintf(fp,"<free>"); }
         else if( m_allocation == READ_ALLOC ) { fprintf(fp,"rd: "); m_op.dump(fp); }
         else if( m_allocation == WRITE_ALLOC ) { fprintf(fp,"wr: "); m_op.dump(fp); }
         fprintf(fp,"\n");
      }
      void alloc_read( const op_t &op )  { assert(is_free()); m_allocation=READ_ALLOC; m_op=op; }
      void alloc_write( const op_t &op , unsigned subbank, int warp_ID = -1 ,int reg = -1) {
    	  //unsigned subbank;

    	  assert(is_free()); m_allocation=WRITE_ALLOC; m_op=op;

    	  //DQ_code//
    	  assert(sub_is_free(subbank)); sub_m_allocation[subbank]=WRITE_ALLOC;



    	  /////DQ_code///////////
    	 /* if ((warp_ID !=-1 ) & (reg != -1) ){
    	  NV = flag[warp_ID][reg];
          warp_ID_out = warp_ID;
          reg_out = reg;
    	  }*/



      }
      void reset() { m_allocation = NO_ALLOC; }
      void reset_sub(unsigned subbank){sub_m_allocation[subbank] = NO_ALLOC;}


   private:

      int NV;
      int warp_ID_out;
      int reg_out;
      int flag [32] [32];/////32= warp ;32= register/warp;
      int NV_Nu[16];
      int SRAM_Nu[16];
      int Num_NV;
      int Num_Sram;
      //unsigned subbank_size_al;

      //DQ_code//configuration of subbank
      //int subbank_size;
      enum alloc_t m_allocation;
      //DQ_code//
      enum alloc_t* sub_m_allocation;
      int  sub_config;
      op_t m_op;

      /////DQ_Code/////////

   };

   class arbiter_t {
   public:
      // constructors
      arbiter_t()
      {
         m_queue=NULL;
         m_allocated_bank=NULL;
         m_allocator_rr_head=NULL;
         _inmatch=NULL;
         _outmatch=NULL;
         _request=NULL;
         m_last_cu=0;
         //DQ_code
         sub_config=200;

         sub_size=16;
         //for(int i = 0 ; i<32; ++i)
        //{counter[i] = 0;}////DQ_CODE//////////

         ///DQ_code////
         /*for(int i = 0; i<m_num_banks; ++i){
        	 for(int j = 0; j<8192; ++j){
        		 subbank[i][j]=0;
        	 }

         }*/
      }
      void init( unsigned num_cu, unsigned num_banks, shader_core_ctx* shader )
      { 
         assert(num_cu > 0);
         assert(num_banks > 0);
         //m_shader = new shader_core_ctx;
         m_shader = shader;
         m_num_collectors = num_cu;
         m_num_banks = num_banks;
         _inmatch = new int[ m_num_banks ];
         _outmatch = new int[ m_num_collectors ];
         _request = new int*[ m_num_banks ];
         for(unsigned i=0; i<m_num_banks;i++) 
             _request[i] = new int[m_num_collectors];
         m_queue = new std::list<op_t>[num_banks];
         m_allocated_bank = new allocation_t[num_banks];
         m_allocator_rr_head = new unsigned[num_cu];
         for( unsigned n=0; n<num_cu;n++ ) 
            m_allocator_rr_head[n] = n%num_banks;
         counter= new int*[sub_config];
         for(int i =0;i<sub_config;i++){
        	 counter[i]=new int[16];
         }
       //  counter = 0;/////DQ_CODE//////////
         for(int i = 0 ; i<16; ++i)
         {
        	 for(int j = 0; j < sub_config; j++){
        		 counter[j][i] = 0;
        	 }
         }
      //   reset_alloction();
         for( unsigned b=0; b < m_num_banks; b++ )
                {
        	 m_allocated_bank[b].reset();
                }

      }

      // accessors
      void dump(FILE *fp) const
      {
         fprintf(fp,"\n");
         fprintf(fp,"  Arbiter State:\n");
         fprintf(fp,"  requests:\n");
         for( unsigned b=0; b<m_num_banks; b++ ) {
            fprintf(fp,"    bank %u : ", b );
            std::list<op_t>::const_iterator o = m_queue[b].begin();
            for(; o != m_queue[b].end(); o++ ) {
               o->dump(fp);
            }
            fprintf(fp,"\n");
         }
         fprintf(fp,"  grants:\n");
         for(unsigned b=0;b<m_num_banks;b++) {
            fprintf(fp,"    bank %u : ", b );
            m_allocated_bank[b].dump(fp);
         }
         fprintf(fp,"\n");
      }

      // modifiers
      std::list<op_t> allocate_reads(); 

      void add_read_requests( collector_unit_t *cu ) 
      {
         const op_t *src = cu->get_operands();
         for( unsigned i=0; i<MAX_REG_OPERANDS*2; i++) {
            const op_t &op = src[i];
            if( op.valid() ) {
               unsigned bank = op.get_bank();
               m_queue[bank].push_back(op);
            }
         }
      }
      bool bank_idle( unsigned bank ) const
      {
          return m_allocated_bank[bank].is_free();
      }

      //DQ_code//
      bool Subbank_idle (unsigned bank, unsigned subbank)
      {
    	  return m_allocated_bank[bank].sub_is_free(subbank);
      }


      void allocate_bank_for_write( unsigned bank, const op_t &op , unsigned subbank)
      {
    	// printf("bank:%u____m_num_banks:%u\n",bank,m_num_banks);
         assert( bank < m_num_banks );
         m_allocated_bank[bank].alloc_write(op, subbank);
         //DQ_code//

      }
      void allocate_for_read( unsigned bank, const op_t &op )
      {
         assert( bank < m_num_banks );
         m_allocated_bank[bank].alloc_read(op);
      }
      void reset_alloction();


   private:
      unsigned m_num_banks;
      unsigned m_num_collectors;
      int sub_config;

      int** counter;//[200][32];  ///////DQ_code//////
      int flag;


      allocation_t *m_allocated_bank; // bank # -> register that wins
      std::list<op_t> *m_queue;

      unsigned *m_allocator_rr_head; // cu # -> next bank to check for request (rr-arb)
      unsigned  m_last_cu; // first cu to check while arb-ing banks (rr)

      int *_inmatch;
      int *_outmatch;
      int **_request;
      shader_core_ctx *m_shader;
      unsigned sub_size;
      //DQ_code//
      //int bankSize=16;//configuration
      //int partition=8;
      //int subbank[8192][16];
   };

   class input_port_t {
   public:
       input_port_t(port_vector_t & input, port_vector_t & output, uint_vector_t cu_sets)
       : m_in(input),m_out(output), m_cu_sets(cu_sets)
       {
           assert(input.size() == output.size());
           assert(not m_cu_sets.empty());
       }
   //private:
       port_vector_t m_in,m_out;
       uint_vector_t m_cu_sets;
   };

   class collector_unit_t {
   public:
      // constructors
      collector_unit_t()
      { 
         m_free = true;
         m_warp = NULL;
         m_output_register = NULL;
         m_src_op = new op_t[MAX_REG_OPERANDS*2];
         m_not_ready.reset();
         m_warp_id = -1;
         m_num_banks = 0;
         m_bank_warp_shift = 0;
      }
      // accessors
      bool ready() const;
      const op_t *get_operands() const { return m_src_op; }
      void dump(FILE *fp, const shader_core_ctx *shader ) const;

      unsigned get_warp_id() const { return m_warp_id; }
      unsigned get_active_count() const { return m_warp->active_count(); }
      const active_mask_t & get_active_mask() const { return m_warp->get_active_mask(); }
      unsigned get_sp_op() const { return m_warp->sp_op; }
      unsigned get_id() const { return m_cuid; } // returns CU hw id

      // modifiers
      void init(unsigned n, 
                unsigned num_banks, 
                unsigned log2_warp_size,
                const core_config *config,
                opndcoll_rfu_t *rfu ); 
      bool allocate( register_set* pipeline_reg, register_set* output_reg );

      void collect_operand( unsigned op )
      {
         m_not_ready.reset(op);
      }
      unsigned get_num_operands() const{
    	  return m_warp->get_num_operands();
      }
      unsigned get_num_regs() const{
    	  return m_warp->get_num_regs();
      }
      void dispatch();
      bool is_free(){return m_free;}

   private:
      bool m_free;
      unsigned m_cuid; // collector unit hw id
      unsigned m_warp_id;
      warp_inst_t  *m_warp;
      register_set* m_output_register; // pipeline register to issue to when ready
      op_t *m_src_op;
      std::bitset<MAX_REG_OPERANDS*2> m_not_ready;
      unsigned m_num_banks;
      unsigned m_bank_warp_shift;
      opndcoll_rfu_t *m_rfu;

   };

   class dispatch_unit_t {
   public:
      dispatch_unit_t(std::vector<collector_unit_t>* cus) 
      { 
         m_last_cu=0;
         m_collector_units=cus;
         m_num_collectors = (*cus).size();
         m_next_cu=0;
      }

      collector_unit_t *find_ready()
      {
         for( unsigned n=0; n < m_num_collectors; n++ ) {
            unsigned c=(m_last_cu+n+1)%m_num_collectors;////////+N/////////DQ//////////
            if( (*m_collector_units)[c].ready() ) {
               m_last_cu=c;
               return &((*m_collector_units)[c]);
            }
         }
         return NULL;
      }

   private:
      unsigned m_num_collectors;
      std::vector<collector_unit_t>* m_collector_units;
      unsigned m_last_cu; // dispatch ready cu's rr
      unsigned m_next_cu;  // for initialization
   };

   // opndcoll_rfu_t data members
   bool m_initialized;

   unsigned m_num_collector_sets;
   //unsigned m_num_collectors;
   unsigned m_num_banks;
   unsigned m_bank_warp_shift;
   unsigned m_warp_size;
   std::vector<collector_unit_t *> m_cu;
   arbiter_t m_arbiter;

   //unsigned m_num_ports;
   //std::vector<warp_inst_t**> m_input;
   //std::vector<warp_inst_t**> m_output;
   //std::vector<unsigned> m_num_collector_units;
   //warp_inst_t **m_alu_port;

   std::vector<input_port_t> m_in_ports;
   typedef std::map<unsigned /* collector set */, std::vector<collector_unit_t> /*collector sets*/ > cu_sets_t;
   cu_sets_t m_cus;
   std::vector<dispatch_unit_t> m_dispatch_units;

   //typedef std::map<warp_inst_t**/*port*/,dispatch_unit_t> port_to_du_t;
   //port_to_du_t                     m_dispatch_units;
   //std::map<warp_inst_t**,std::list<collector_unit_t*> > m_free_cu;
   shader_core_ctx* m_shader;


   ////DQ_code///flag///
   //unsigned valid[8192] [4];
   int sub_config;
   unsigned Nbank;
   unsigned sub_size;
   unsigned** count;//[200] [4];
   int inc;
   unsigned row_nu;
   unsigned bank;
   int** flag;// [200] [4];///DQ_code////DQ_config///
   int**  row_flag;//[200] [4];
   int**  bank_flag;//[200] [4];
   int flagCount;
   unsigned count_delay[16];///DQ_code////DQ_config/////
   //unsigned m_subbank;

};

class barrier_set_t {
public:
   barrier_set_t(shader_core_ctx * shader, unsigned max_warps_per_core, unsigned max_cta_per_core, unsigned max_barriers_per_cta, unsigned warp_size);

   // during cta allocation
   void allocate_barrier( unsigned cta_id, warp_set_t warps );

   // during cta deallocation
   void deallocate_barrier( unsigned cta_id );

   typedef std::map<unsigned, warp_set_t >  cta_to_warp_t;
   typedef std::map<unsigned, warp_set_t >  bar_id_to_warp_t; /*set of warps reached a specific barrier id*/


   // individual warp hits barrier
   void warp_reaches_barrier( unsigned cta_id, unsigned warp_id, warp_inst_t* inst);


   // warp reaches exit 
   void warp_exit( unsigned warp_id );

   // assertions
   bool warp_waiting_at_barrier( unsigned warp_id ) const;

   // debug
   void dump();

private:
   unsigned m_max_cta_per_core;
   unsigned m_max_warps_per_core;
   unsigned m_max_barriers_per_cta;
   unsigned m_warp_size;
   cta_to_warp_t m_cta_to_warps;
   bar_id_to_warp_t m_bar_id_to_warps;
   warp_set_t m_warp_active;
   warp_set_t m_warp_at_barrier;
   shader_core_ctx *m_shader;

};

struct insn_latency_info {
   unsigned pc;
   unsigned long latency;
};

struct ifetch_buffer_t {
    ifetch_buffer_t() { m_valid=false; }

    ifetch_buffer_t( address_type pc, unsigned nbytes, unsigned warp_id ) 
    { 
        m_valid=true; 
        m_pc=pc; 
        m_nbytes=nbytes; 
        m_warp_id=warp_id;
    }

    bool m_valid;
    address_type m_pc;
    unsigned m_nbytes;
    unsigned m_warp_id;
};

class shader_core_config;

class simd_function_unit {
public:
    simd_function_unit( const shader_core_config *config );
    ~simd_function_unit() { delete m_dispatch_reg; }

    // modifiers
    virtual void issue( register_set& source_reg ) { source_reg.move_out_to(m_dispatch_reg); occupied.set(m_dispatch_reg->latency);}
    virtual void cycle() = 0;
    virtual void active_lanes_in_pipeline() = 0;

    // accessors
    virtual unsigned clock_multiplier() const { return 1; }
    virtual bool can_issue( const warp_inst_t &inst ) const { return m_dispatch_reg->empty() && !occupied.test(inst.latency); }
    virtual bool stallable() const = 0;
    virtual void print( FILE *fp ) const
    {
        fprintf(fp,"%s dispatch= ", m_name.c_str() );
        m_dispatch_reg->print(fp);
    }
protected:
    std::string m_name;
    const shader_core_config *m_config;
    warp_inst_t *m_dispatch_reg;
    static const unsigned MAX_ALU_LATENCY = 512;
    std::bitset<MAX_ALU_LATENCY> occupied;
};

class pipelined_simd_unit : public simd_function_unit {
public:
    pipelined_simd_unit( register_set* result_port, const shader_core_config *config, unsigned max_latency, shader_core_ctx *core );

    //modifiers
    virtual void cycle();
    virtual void issue( register_set& source_reg );
    virtual unsigned get_active_lanes_in_pipeline()
    {
    	active_mask_t active_lanes;
    	active_lanes.reset();
        for( unsigned stage=0; (stage+1)<m_pipeline_depth; stage++ ){
        	if( !m_pipeline_reg[stage]->empty() )
        		active_lanes|=m_pipeline_reg[stage]->get_active_mask();
        }
        return active_lanes.count();
    }
    virtual void active_lanes_in_pipeline() = 0;
/*
    virtual void issue( register_set& source_reg )
    {
        //move_warp(m_dispatch_reg,source_reg);
        //source_reg.move_out_to(m_dispatch_reg);
        simd_function_unit::issue(source_reg);
    }
*/
    // accessors
    virtual bool stallable() const { return false; }
    virtual bool can_issue( const warp_inst_t &inst ) const
    {
        return simd_function_unit::can_issue(inst);
    }
    virtual void print(FILE *fp) const
    {
        simd_function_unit::print(fp);
        for( int s=m_pipeline_depth-1; s>=0; s-- ) {
            if( !m_pipeline_reg[s]->empty() ) { 
                fprintf(fp,"      %s[%2d] ", m_name.c_str(), s );
                m_pipeline_reg[s]->print(fp);
            }
        }
    }
protected:
    unsigned m_pipeline_depth;
    warp_inst_t **m_pipeline_reg;
    register_set *m_result_port;
    class shader_core_ctx *m_core;
};

class sfu : public pipelined_simd_unit
{
public:
    sfu( register_set* result_port, const shader_core_config *config, shader_core_ctx *core );
    virtual bool can_issue( const warp_inst_t &inst ) const
    {
        switch(inst.op) {
        case SFU_OP: break;
        case ALU_SFU_OP: break;
        default: return false;
        }
        return pipelined_simd_unit::can_issue(inst);
    }
    virtual void active_lanes_in_pipeline();
    virtual void issue(  register_set& source_reg );
};

class sp_unit : public pipelined_simd_unit
{
public:
    sp_unit( register_set* result_port, const shader_core_config *config, shader_core_ctx *core );
    virtual bool can_issue( const warp_inst_t &inst ) const
    {
        switch(inst.op) {
        case SFU_OP: return false; 
        case LOAD_OP: return false;
        case STORE_OP: return false;
        case MEMORY_BARRIER_OP: return false;
        default: break;
        }
        return pipelined_simd_unit::can_issue(inst);
    }
    virtual void active_lanes_in_pipeline();
    virtual void issue( register_set& source_reg );
};

class simt_core_cluster;
class shader_memory_interface;
class shader_core_mem_fetch_allocator;
class cache_t;

class ldst_unit: public pipelined_simd_unit {
public:
    ldst_unit( mem_fetch_interface *icnt,
               shader_core_mem_fetch_allocator *mf_allocator,
               shader_core_ctx *core, 
               opndcoll_rfu_t *operand_collector,
               Scoreboard *scoreboard,
               const shader_core_config *config, 
               const memory_config *mem_config,  
               class shader_core_stats *stats, 
               unsigned sid, unsigned tpc );

    // modifiers
    virtual void issue( register_set &inst );
    virtual void cycle();
     
    void fill( mem_fetch *mf );
    void flush();
    void writeback();

    // accessors
    virtual unsigned clock_multiplier() const;

    virtual bool can_issue( const warp_inst_t &inst ) const
    {
        switch(inst.op) {
        case LOAD_OP: break;
        case STORE_OP: break;
        case MEMORY_BARRIER_OP: break;
        default: return false;
        }
        return m_dispatch_reg->empty();
    }

    virtual void active_lanes_in_pipeline();
    virtual bool stallable() const { return true; }
    bool response_buffer_full() const;
    void print(FILE *fout) const;
    void print_cache_stats( FILE *fp, unsigned& dl1_accesses, unsigned& dl1_misses );
    void get_cache_stats(unsigned &read_accesses, unsigned &write_accesses, unsigned &read_misses, unsigned &write_misses, unsigned cache_type);
    void get_cache_stats(cache_stats &cs);

    void get_L1D_sub_stats(struct cache_sub_stats &css) const;
    void get_L1C_sub_stats(struct cache_sub_stats &css) const;
    void get_L1T_sub_stats(struct cache_sub_stats &css) const;

protected:
    ldst_unit( mem_fetch_interface *icnt,
               shader_core_mem_fetch_allocator *mf_allocator,
               shader_core_ctx *core, 
               opndcoll_rfu_t *operand_collector,
               Scoreboard *scoreboard,
               const shader_core_config *config,
               const memory_config *mem_config,  
               shader_core_stats *stats,
               unsigned sid,
               unsigned tpc,
               l1_cache* new_l1d_cache );
    void init( mem_fetch_interface *icnt,
               shader_core_mem_fetch_allocator *mf_allocator,
               shader_core_ctx *core, 
               opndcoll_rfu_t *operand_collector,
               Scoreboard *scoreboard,
               const shader_core_config *config,
               const memory_config *mem_config,  
               shader_core_stats *stats,
               unsigned sid,
               unsigned tpc );

protected:
   bool shared_cycle( warp_inst_t &inst, mem_stage_stall_type &rc_fail, mem_stage_access_type &fail_type);
   bool constant_cycle( warp_inst_t &inst, mem_stage_stall_type &rc_fail, mem_stage_access_type &fail_type);
   bool texture_cycle( warp_inst_t &inst, mem_stage_stall_type &rc_fail, mem_stage_access_type &fail_type);
   bool memory_cycle( warp_inst_t &inst, mem_stage_stall_type &rc_fail, mem_stage_access_type &fail_type);

   virtual mem_stage_stall_type process_cache_access( cache_t* cache,
                                                      new_addr_type address,
                                                      warp_inst_t &inst,
                                                      std::list<cache_event>& events,
                                                      mem_fetch *mf,
                                                      enum cache_request_status status );
   mem_stage_stall_type process_memory_access_queue( cache_t *cache, warp_inst_t &inst );

   const memory_config *m_memory_config;
   class mem_fetch_interface *m_icnt;
   shader_core_mem_fetch_allocator *m_mf_allocator;
   class shader_core_ctx *m_core;
   unsigned m_sid;
   unsigned m_tpc;

   tex_cache *m_L1T; // texture cache
   read_only_cache *m_L1C; // constant cache
   l1_cache *m_L1D; // data cache
   std::map<unsigned/*warp_id*/, std::map<unsigned/*regnum*/,unsigned/*count*/> > m_pending_writes;
   std::list<mem_fetch*> m_response_fifo;
   opndcoll_rfu_t *m_operand_collector;
   Scoreboard *m_scoreboard;

   mem_fetch *m_next_global;
   warp_inst_t m_next_wb;
   unsigned m_writeback_arb; // round-robin arbiter for writeback contention between L1T, L1C, shared
   unsigned m_num_writeback_clients;

   enum mem_stage_stall_type m_mem_rc;

   shader_core_stats *m_stats; 

   // for debugging
   unsigned long long m_last_inst_gpu_sim_cycle;
   unsigned long long m_last_inst_gpu_tot_sim_cycle;
};

enum pipeline_stage_name_t {
    ID_OC_SP=0,
    ID_OC_SFU,  
    ID_OC_MEM,  
    OC_EX_SP,
    OC_EX_SFU,
    OC_EX_MEM,
    EX_WB,
    N_PIPELINE_STAGES 
};

const char* const pipeline_stage_name_decode[] = {
    "ID_OC_SP",
    "ID_OC_SFU",  
    "ID_OC_MEM",  
    "OC_EX_SP",
    "OC_EX_SFU",
    "OC_EX_MEM",
    "EX_WB",
    "N_PIPELINE_STAGES" 
};

struct shader_core_config : public core_config
{
    shader_core_config(){
	pipeline_widths_string = NULL;
    }

    void init()
    {
        int ntok = sscanf(gpgpu_shader_core_pipeline_opt,"%d:%d", 
                          &n_thread_per_shader,
                          &warp_size);
        if(ntok != 2) {
           printf("GPGPU-Sim uArch: error while parsing configuration string gpgpu_shader_core_pipeline_opt\n");
           abort();
	}

	char* toks = new char[100];
	char* tokd = toks;
	strcpy(toks,pipeline_widths_string);

	toks = strtok(toks,",");
	for (unsigned i = 0; i < N_PIPELINE_STAGES; i++) { 
	    assert(toks);
	    ntok = sscanf(toks,"%d", &pipe_widths[i]);
	    assert(ntok == 1); 
	    toks = strtok(NULL,",");
	}
	delete[] tokd;

        if (n_thread_per_shader > MAX_THREAD_PER_SM) {
           printf("GPGPU-Sim uArch: Error ** increase MAX_THREAD_PER_SM in abstract_hardware_model.h from %u to %u\n", 
                  MAX_THREAD_PER_SM, n_thread_per_shader);
           abort();
        }
        max_warps_per_shader =  n_thread_per_shader/warp_size;
        assert( !(n_thread_per_shader % warp_size) );
        max_sfu_latency = 512;
        max_sp_latency = 32;
        m_L1I_config.init(m_L1I_config.m_config_string,FuncCachePreferNone);
        m_L1T_config.init(m_L1T_config.m_config_string,FuncCachePreferNone);
        m_L1C_config.init(m_L1C_config.m_config_string,FuncCachePreferNone);
        m_L1D_config.init(m_L1D_config.m_config_string,FuncCachePreferNone);
        gpgpu_cache_texl1_linesize = m_L1T_config.get_line_sz();
        gpgpu_cache_constl1_linesize = m_L1C_config.get_line_sz();
        m_valid = true;
    }
    void reg_options(class OptionParser * opp );
    unsigned max_cta( const kernel_info_t &k ) const;
    unsigned num_shader() const { return n_simt_clusters*n_simt_cores_per_cluster; }
    unsigned sid_to_cluster( unsigned sid ) const { return sid / n_simt_cores_per_cluster; }
    unsigned sid_to_cid( unsigned sid )     const { return sid % n_simt_cores_per_cluster; }
    unsigned cid_to_sid( unsigned cid, unsigned cluster_id ) const { return cluster_id*n_simt_cores_per_cluster + cid; }

// data
    char *gpgpu_shader_core_pipeline_opt;
    bool gpgpu_perfect_mem;
    bool gpgpu_clock_gated_reg_file;
    bool gpgpu_clock_gated_lanes;
    enum divergence_support_t model;
    unsigned n_thread_per_shader;
    unsigned n_regfile_gating_group;
    unsigned max_warps_per_shader; 
    unsigned max_cta_per_core; //Limit on number of concurrent CTAs in shader core
    unsigned max_barriers_per_cta;
    char * gpgpu_scheduler_string;

    char* pipeline_widths_string;
    int pipe_widths[N_PIPELINE_STAGES];

    mutable cache_config m_L1I_config;
    mutable cache_config m_L1T_config;
    mutable cache_config m_L1C_config;
    mutable l1d_cache_config m_L1D_config;

    bool gmem_skip_L1D; // on = global memory access always skip the L1 cache 
    
    bool gpgpu_dwf_reg_bankconflict;

    int gpgpu_num_sched_per_core;
    int gpgpu_max_insn_issue_per_warp;

    //op collector
    int gpgpu_operand_collector_num_units_sp;
    int gpgpu_operand_collector_num_units_sfu;
    int gpgpu_operand_collector_num_units_mem;
    int gpgpu_operand_collector_num_units_gen;

    unsigned int gpgpu_operand_collector_num_in_ports_sp;
    unsigned int gpgpu_operand_collector_num_in_ports_sfu;
    unsigned int gpgpu_operand_collector_num_in_ports_mem;
    unsigned int gpgpu_operand_collector_num_in_ports_gen;

    unsigned int gpgpu_operand_collector_num_out_ports_sp;
    unsigned int gpgpu_operand_collector_num_out_ports_sfu;
    unsigned int gpgpu_operand_collector_num_out_ports_mem;
    unsigned int gpgpu_operand_collector_num_out_ports_gen;

    int gpgpu_num_sp_units;
    int gpgpu_num_sfu_units;
    int gpgpu_num_mem_units;

    //Shader core resources
    unsigned gpgpu_shader_registers;
    int gpgpu_warpdistro_shader;
    int gpgpu_warp_issue_shader;
    unsigned gpgpu_num_reg_banks;
    bool gpgpu_reg_bank_use_warp_id;
    bool gpgpu_local_mem_map;
    
    unsigned max_sp_latency;
    unsigned max_sfu_latency;
    
    unsigned n_simt_cores_per_cluster;
    unsigned n_simt_clusters;
    unsigned n_simt_ejection_buffer_size;
    unsigned ldst_unit_response_queue_size;

    int simt_core_sim_order; 
    
    unsigned mem2device(unsigned memid) const { return memid + n_simt_clusters; }
};

struct shader_core_stats_pod {

	void* shader_core_stats_pod_start[0]; // DO NOT MOVE FROM THE TOP - spaceless pointer to the start of this structure
	unsigned long long *shader_cycles;
    unsigned *m_num_sim_insn; // number of scalar thread instructions committed by this shader core
    unsigned *m_num_sim_winsn; // number of warp instructions committed by this shader core
	unsigned *m_last_num_sim_insn;
	unsigned *m_last_num_sim_winsn;
    unsigned *m_num_decoded_insn; // number of instructions decoded by this shader core
    float *m_pipeline_duty_cycle;
    unsigned *m_num_FPdecoded_insn;
    unsigned *m_num_INTdecoded_insn;
    unsigned *m_num_storequeued_insn;
    unsigned *m_num_loadqueued_insn;
    unsigned *m_num_ialu_acesses;
    unsigned *m_num_fp_acesses;
    unsigned *m_num_imul_acesses;
    unsigned *m_num_tex_inst;
    unsigned *m_num_fpmul_acesses;
    unsigned *m_num_idiv_acesses;
    unsigned *m_num_fpdiv_acesses;
    unsigned *m_num_sp_acesses;
    unsigned *m_num_sfu_acesses;
    unsigned *m_num_trans_acesses;
    unsigned *m_num_mem_acesses;
    unsigned *m_num_sp_committed;
    unsigned *m_num_tlb_hits;
    unsigned *m_num_tlb_accesses;
    unsigned *m_num_sfu_committed;
    unsigned *m_num_mem_committed;
    unsigned *m_read_regfile_acesses;
    unsigned *m_write_regfile_acesses;
    unsigned *m_non_rf_operands;
    unsigned *m_num_imul24_acesses;
    unsigned *m_num_imul32_acesses;
    unsigned *m_active_sp_lanes;
    unsigned *m_active_sfu_lanes;
    unsigned *m_active_fu_lanes;
    unsigned *m_active_fu_mem_lanes;
    unsigned *m_n_diverge;    // number of divergence occurring in this shader
    unsigned gpgpu_n_load_insn;
    unsigned gpgpu_n_store_insn;
    unsigned gpgpu_n_shmem_insn;
    unsigned gpgpu_n_tex_insn;
    unsigned gpgpu_n_const_insn;
    unsigned gpgpu_n_param_insn;
    unsigned gpgpu_n_shmem_bkconflict;
    unsigned gpgpu_n_cache_bkconflict;
    int      gpgpu_n_intrawarp_mshr_merge;
    unsigned gpgpu_n_cmem_portconflict;
    unsigned gpu_stall_shd_mem_breakdown[N_MEM_STAGE_ACCESS_TYPE][N_MEM_STAGE_STALL_TYPE];
    unsigned gpu_reg_bank_conflict_stalls;
    unsigned *shader_cycle_distro;
    unsigned *last_shader_cycle_distro;
    unsigned *num_warps_issuable;
    unsigned gpgpu_n_stall_shd_mem;

    //memory access classification
    int gpgpu_n_mem_read_local;
    int gpgpu_n_mem_write_local;
    int gpgpu_n_mem_texture;
    int gpgpu_n_mem_const;
    int gpgpu_n_mem_read_global;
    int gpgpu_n_mem_write_global;
    int gpgpu_n_mem_read_inst;
    
    int gpgpu_n_mem_l2_writeback;
    int gpgpu_n_mem_l1_write_allocate; 
    int gpgpu_n_mem_l2_write_allocate;

    unsigned made_write_mfs;
    unsigned made_read_mfs;

    unsigned *gpgpu_n_shmem_bank_access;
    long *n_simt_to_mem; // Interconnect power stats
    long *n_mem_to_simt;
};

class shader_core_stats : public shader_core_stats_pod {
public:
    shader_core_stats( const shader_core_config *config )
    {
        m_config = config;
        shader_core_stats_pod *pod = reinterpret_cast< shader_core_stats_pod * > ( this->shader_core_stats_pod_start );
        memset(pod,0,sizeof(shader_core_stats_pod));
        shader_cycles=(unsigned long long *) calloc(config->num_shader(),sizeof(unsigned long long ));
        m_num_sim_insn = (unsigned*) calloc(config->num_shader(),sizeof(unsigned));
        m_num_sim_winsn = (unsigned*) calloc(config->num_shader(),sizeof(unsigned));
        m_last_num_sim_winsn = (unsigned*) calloc(config->num_shader(),sizeof(unsigned));
        m_last_num_sim_insn = (unsigned*) calloc(config->num_shader(),sizeof(unsigned));
        m_pipeline_duty_cycle=(float*) calloc(config->num_shader(),sizeof(float));
        m_num_decoded_insn = (unsigned*) calloc(config->num_shader(),sizeof(unsigned));
        m_num_FPdecoded_insn = (unsigned*) calloc(config->num_shader(),sizeof(unsigned));
        m_num_storequeued_insn=(unsigned*) calloc(config->num_shader(),sizeof(unsigned));
        m_num_loadqueued_insn=(unsigned*) calloc(config->num_shader(),sizeof(unsigned));
        m_num_INTdecoded_insn = (unsigned*) calloc(config->num_shader(),sizeof(unsigned));
        m_num_ialu_acesses = (unsigned*) calloc(config->num_shader(),sizeof(unsigned));
        m_num_fp_acesses= (unsigned*) calloc(config->num_shader(),sizeof(unsigned));
        m_num_tex_inst= (unsigned*) calloc(config->num_shader(),sizeof(unsigned));
        m_num_imul_acesses= (unsigned*) calloc(config->num_shader(),sizeof(unsigned));
        m_num_imul24_acesses= (unsigned*) calloc(config->num_shader(),sizeof(unsigned));
        m_num_imul32_acesses= (unsigned*) calloc(config->num_shader(),sizeof(unsigned));
        m_num_fpmul_acesses= (unsigned*) calloc(config->num_shader(),sizeof(unsigned));
        m_num_idiv_acesses= (unsigned*) calloc(config->num_shader(),sizeof(unsigned));
        m_num_fpdiv_acesses= (unsigned*) calloc(config->num_shader(),sizeof(unsigned));
        m_num_sp_acesses= (unsigned*) calloc(config->num_shader(),sizeof(unsigned));
        m_num_sfu_acesses= (unsigned*) calloc(config->num_shader(),sizeof(unsigned));
        m_num_trans_acesses= (unsigned*) calloc(config->num_shader(),sizeof(unsigned));
        m_num_mem_acesses= (unsigned*) calloc(config->num_shader(),sizeof(unsigned));
        m_num_sp_committed= (unsigned*) calloc(config->num_shader(),sizeof(unsigned));
        m_num_tlb_hits=(unsigned*) calloc(config->num_shader(),sizeof(unsigned));
        m_num_tlb_accesses=(unsigned*) calloc(config->num_shader(),sizeof(unsigned));
        m_active_sp_lanes= (unsigned*) calloc(config->num_shader(),sizeof(unsigned));
        m_active_sfu_lanes= (unsigned*) calloc(config->num_shader(),sizeof(unsigned));
        m_active_fu_lanes= (unsigned*) calloc(config->num_shader(),sizeof(unsigned));
        m_active_fu_mem_lanes= (unsigned*) calloc(config->num_shader(),sizeof(unsigned));
        m_num_sfu_committed= (unsigned*) calloc(config->num_shader(),sizeof(unsigned));
        m_num_mem_committed= (unsigned*) calloc(config->num_shader(),sizeof(unsigned));
        m_read_regfile_acesses= (unsigned*) calloc(config->num_shader(),sizeof(unsigned));
        m_write_regfile_acesses= (unsigned*) calloc(config->num_shader(),sizeof(unsigned));
        m_non_rf_operands=(unsigned*) calloc(config->num_shader(),sizeof(unsigned));
        m_n_diverge = (unsigned*) calloc(config->num_shader(),sizeof(unsigned));
        shader_cycle_distro = (unsigned*) calloc(config->warp_size+3, sizeof(unsigned));
        last_shader_cycle_distro = (unsigned*) calloc(m_config->warp_size+3, sizeof(unsigned));

        n_simt_to_mem = (long *)calloc(config->num_shader(), sizeof(long));
        n_mem_to_simt = (long *)calloc(config->num_shader(), sizeof(long));

        m_outgoing_traffic_stats = new traffic_breakdown("coretomem"); 
        m_incoming_traffic_stats = new traffic_breakdown("memtocore"); 

        gpgpu_n_shmem_bank_access = (unsigned *)calloc(config->num_shader(), sizeof(unsigned));

        m_shader_dynamic_warp_issue_distro.resize( config->num_shader() );
        m_shader_warp_slot_issue_distro.resize( config->num_shader() );



    }

    ~shader_core_stats()
    {
        delete m_outgoing_traffic_stats; 
        delete m_incoming_traffic_stats; 
        free(m_num_sim_insn); 
        free(m_num_sim_winsn);
        free(m_n_diverge); 
        free(shader_cycle_distro);
        free(last_shader_cycle_distro);
    }

    void new_grid()
    {
    }

    void event_warp_issued( unsigned s_id, unsigned warp_id, unsigned num_issued, unsigned dynamic_warp_id );

    void visualizer_print( gzFile visualizer_file );

    void print( FILE *fout ) const;

    const std::vector< std::vector<unsigned> >& get_dynamic_warp_issue() const
    {
        return m_shader_dynamic_warp_issue_distro;
    }

    const std::vector< std::vector<unsigned> >& get_warp_slot_issue() const
    {
        return m_shader_warp_slot_issue_distro;
    }

private:
    const shader_core_config *m_config;

    traffic_breakdown *m_outgoing_traffic_stats; // core to memory partitions
    traffic_breakdown *m_incoming_traffic_stats; // memory partition to core 

    // Counts the instructions issued for each dynamic warp.
    std::vector< std::vector<unsigned> > m_shader_dynamic_warp_issue_distro;
    std::vector<unsigned> m_last_shader_dynamic_warp_issue_distro;
    std::vector< std::vector<unsigned> > m_shader_warp_slot_issue_distro;
    std::vector<unsigned> m_last_shader_warp_slot_issue_distro;

    friend class power_stat_t;
    friend class shader_core_ctx;
    friend class ldst_unit;
    friend class simt_core_cluster;
    friend class scheduler_unit;
    friend class TwoLevelScheduler;
    friend class LooseRoundRobbinScheduler;
};

class shader_core_mem_fetch_allocator : public mem_fetch_allocator {
public:
    shader_core_mem_fetch_allocator( unsigned core_id, unsigned cluster_id, const memory_config *config )
    {
    	m_core_id = core_id;
    	m_cluster_id = cluster_id;
    	m_memory_config = config;
    }
    mem_fetch *alloc( new_addr_type addr, mem_access_type type, unsigned size, bool wr ) const 
    {
    	mem_access_t access( type, addr, size, wr );
    	mem_fetch *mf = new mem_fetch( access, 
    				       NULL,
    				       wr?WRITE_PACKET_SIZE:READ_PACKET_SIZE, 
    				       -1, 
    				       m_core_id, 
    				       m_cluster_id,
    				       m_memory_config );
    	return mf;
    }
    
    mem_fetch *alloc( const warp_inst_t &inst, const mem_access_t &access ) const
    {
        warp_inst_t inst_copy = inst;
        mem_fetch *mf = new mem_fetch(access, 
                                      &inst_copy, 
                                      access.is_write()?WRITE_PACKET_SIZE:READ_PACKET_SIZE,
                                      inst.warp_id(),
                                      m_core_id, 
                                      m_cluster_id, 
                                      m_memory_config);
        return mf;
    }

private:
    unsigned m_core_id;
    unsigned m_cluster_id;
    const memory_config *m_memory_config;
};

class shader_core_ctx : public core_t {
public:

    // creator:
    shader_core_ctx( class gpgpu_sim *gpu,
                     class simt_core_cluster *cluster,
                     unsigned shader_id,
                     unsigned tpc_id,
                     const struct shader_core_config *config,
                     const struct memory_config *mem_config,
                     shader_core_stats *stats );
/////DQ_code///is_flag//////shader_core_ctx///

//    bool is_flag (int row ,int column){return flag[row][column] == 1;}

/////DQ_code///is_flag//////shader_core_ctx///

    void register_disvalid(){
    	if(search==1){
    		register_valid[row_nu][bank]=0;
    	}


    }

    void gpu_pipconflict_count();

    int return_pipOp(){
    	return pipOp;
    }

    int return_interleave(){
    	return interleave;
    }
    //////DQ_code/////////int_flag
    void int_flag(){


    		    for (int i = 0 ; i<sub_config; i++)
    		    {
    		    	for (int j = 0 ; j < 16; j++){
    		    	flag[i][j]=0;
    		    	//sub_tag[i][j]=0;
    		    	}

    		    }
       }

    void gpu_write_upgrade();


    bool m_sid_9(){
    	return (m_sid==9);
    }

    bool sub_tag_compare(unsigned bank, unsigned subbank, unsigned bank_, unsigned row_nu){
    	if(dynamic_mapping==0){
    		if(sub_row_nu_tag[subbank][bank]== row_nu and sub_bank_tag[subbank][bank]== bank_){
				sub_searching=subbank;
				bank_searching=bank;
    			return true;
    		}

    		else
    			return false;
    	}
    	else{
    		assert(dynamic_mapping==1);
    		for(unsigned i=bank; i<16+bank; i++){
    			for(unsigned j = subbank/sub_group*sub_group ; j < subbank/sub_group*sub_group+sub_group; j++){
    				if(sub_row_nu_tag[j][i%16]== row_nu and sub_bank_tag[j][i%16]== bank_){
    					sub_searching=j;
    					bank_searching=i%16;
    					return true;

    				}

    			}

    		}
    		return false;

    	}
    }

    bool flag_search(unsigned bank, unsigned subbank){
    	if(dynamic_mapping==0){

    		//printf("subbank:%u;bank:%u\n;flag:%u",subbank,bank,flag[subbank][bank]);
    		assert(subbank<200);
    		assert(bank<16);
    		if(flag[subbank][bank] !=  1){
    			assert(flag[subbank][bank]==0);
    			//printf("subbank:%d,bank:%d\n",subbank, bank);
    			sub_searching=subbank;
    			bank_searching=bank;
    			return true;
    		}
    		else
    			return false;
    	}
    	else{
    		assert(dynamic_mapping==1);
    		/*if(m_sid==9){
    			    	printf("flag_search:%d\n",flag[subbank][bank]);
    			    	//printf("wid: %d; regid:%d.\n",warp_id,reg);
    			    	//printf("subbank:%d; bank:%d.\n",sub,bank);
    			    	//printf("\n");
    		}*/
    		if(flag[subbank][bank]==0){
    		    					sub_searching=subbank;
    		    					//printf("j:%d",j);
    		    					bank_searching=bank;
    		    					return true;
    		    				}
    		for(unsigned i=bank; i< bank+16;i++){
    			for (unsigned j = subbank/sub_group*sub_group ; j < subbank/sub_group*sub_group+sub_group; j++){
    			//for (unsigned j = subbank/sub_group*sub_group ; j < subbank/sub_group*sub_group+sub_group; j++){
    				if(flag[j][i%16]==0){
    					sub_searching=j;
    					//printf("j:%d",j);
    					bank_searching=i%16;
    					return true;
    				}
    				else{}

    			}

    		}
    		return false;

    	}
    }

    void update_timing(unsigned bank, unsigned subbank){
    	if(dynamic_mapping==0){

    	}
    	else{
    		unsigned n=0;
    		std::vector<unsigned>::iterator s=order_timing_sub.begin();
    		for( std::vector<unsigned>::iterator r=order_timing_bank.begin(); r!=order_timing_bank.end(); r++ ){
    			if(*r==bank and order_timing_sub[n]==subbank){
    				r=order_timing_bank.erase(r);
    				r--;
    				order_timing_sub.erase(order_timing_sub.begin()+n);
    				n--;

    			}
    			n++;

    		}
    		order_timing_bank.push_back(bank);
    		order_timing_sub.push_back(subbank);

    	}

    }

    bool register_valid_zero(unsigned bank, unsigned subbank){
    	for(int i=subbank*subbank_size;i<subbank*subbank_size+subbank_size;i++){
    	    		         			if(register_valid[i][bank]==0){
    	    		         				return true;
    	    		         			}else{}
    	    		         		}
    	return false;
    }

    bool search_oldest(unsigned bank, unsigned sub_nu, unsigned Nbank){
    	if(dynamic_mapping==0){

    	}
    	else{
    		//for(unsigned i=0; i< 16;i++){
    			//for (unsigned j = subbank/sub_group*sub_group ; j < subbank/sub_group*sub_group+sub_group; j++){
    		unsigned n=0;
    		std::vector<unsigned>::iterator s=order_timing_sub.begin();
    		for( std::vector<unsigned>::iterator r=order_timing_bank.begin(); r!=order_timing_bank.end(); r++ ){
    			unsigned temp_bank=*r;
    			unsigned temp_subbank=*(order_timing_sub.begin()+n);
    			n++;
    			if(register_valid_zero(temp_bank,temp_subbank)){
    				if(temp_bank!=Nbank){
    				//for(int i =0; i<){

    				//}
    					if(valid[temp_subbank][temp_bank]==0){
    						sub_searching=temp_subbank;
    						bank_searching=temp_bank;
    						sub_row_nu_tag[temp_subbank][temp_bank]=sub_nu;
    						sub_bank_tag[temp_subbank][temp_bank]=bank;
    						return true;
    					}else{}
    				}
    			}

    		}
    		return false;
    		//return false;	//}
    		//}
    	}


    }
    /*unsigned order_timing(){
    	for(unsigned i =0; i<16;++i){
    		for (unsigned j = subbank/sub_group*sub_group ; j < subbank/sub_group*sub_group+sub_group; j++){
    			if(sub_timing[j][i]>=sub_timing[][]){

    			}
    		}
    	}
    }*/


    unsigned return_bank_remapping(){
    	return bank_remapping;
    }

    unsigned return_sub_nu_remapping(){
        	return sub_nu_remapping;
        }

    void register_available_inc(unsigned bank, unsigned subbank){
    	register_available[subbank][bank]++;
    }

    unsigned return_sub_searching(){
    	return sub_searching;

    }
    unsigned return_bank_searching(){
    	return bank_searching;
        }

    unsigned return_sub_nu(){
    	return sub_nu;
    }
    unsigned return_bank(){
        	return bank;
        }
    unsigned return_dy_sub(){
    	return dynamic_sub;
    }
    unsigned return_dy_bank(){
    	return dynamic_bank;
    }

    bool is_dynamic_mapping(){
    	return (dynamic_mapping==1);
    }
	        //////DQ_code/////////int_flag

    bool return_valid(unsigned bank, unsigned subbank) {
    		return (valid[subbank][bank]==0);
    }
    // used by simt_core_cluster:
    // modifiers

    unsigned return_subbanksize(){
    	return subbank_size;
    }

    void reset_validflag(unsigned bank, unsigned subbank){
    	valid[subbank][bank]=0;
    	flag[subbank][bank]=0;
    }

    void gpu_flag_count();
    ///DQ_code///flag.h
    //DQ_code////write_buffer
    void push_back_reg(std::vector<unsigned> reg){
    	reg_buf.push_back(reg);
    }

    void push_back_bank(std::vector<unsigned> bank){
        	bank_buf.push_back(bank);
        }
    void push_back_inst(warp_inst_t pI){
      	inst_buf.push_back(pI);
      }
    void push_back_free_inst( warp_inst_t inst){
        	free_inst.push_back(inst);
        }

    std::vector<std::vector<unsigned> > reg_out(){
    	return reg_buf;
    }
    std:: vector<std::vector<unsigned> > bank_out(){
    	return bank_buf;
    }
    std::vector<warp_inst_t>inst_out(){
      	return inst_buf;
      }
    void update(std::vector<std::vector<unsigned> > reg,std:: vector<std::vector<unsigned> >bank,std::vector< warp_inst_t>inst){
    	reg_buf.clear();
    	bank_buf.clear();
    	inst_buf.clear();
    	reg_buf = reg;
    	bank_buf = bank;
		inst_buf = inst;
    }



    //void is_flag(const warp_inst_t &pI);
    void is_flag(unsigned reg0,unsigned warp_id0, unsigned Nbank);

    void cycle();
    void reinit(unsigned start_thread, unsigned end_thread, bool reset_not_completed );
    void issue_block2core( class kernel_info_t &kernel );
    void cache_flush();
    void accept_fetch_response( mem_fetch *mf );
    void accept_ldst_unit_response( class mem_fetch * mf );
    void broadcast_barrier_reduction(unsigned cta_id, unsigned bar_id,warp_set_t warps);
    void set_kernel( kernel_info_t *k )
    {
        assert(k);
        m_kernel=k;
        k->inc_running();
        printf("GPGPU-Sim uArch: Shader %d bind to kernel %u \'%s\'\n", m_sid, m_kernel->get_uid(),
                 m_kernel->name().c_str() );
    }

    // accessors
    bool fetch_unit_response_buffer_full() const;
    bool ldst_unit_response_buffer_full() const;
    unsigned get_not_completed() const { return m_not_completed; }
    unsigned get_n_active_cta() const { return m_n_active_cta; }
    unsigned isactive() const {if(m_n_active_cta>0) return 1; else return 0;}
    kernel_info_t *get_kernel() { return m_kernel; }
    unsigned get_sid() const {return m_sid;}

// used by functional simulation:
    // modifiers
    virtual void warp_exit( unsigned warp_id );

    // accessors
    virtual bool warp_waiting_at_barrier( unsigned warp_id ) const;
    void get_pdom_stack_top_info( unsigned tid, unsigned *pc, unsigned *rpc ) const;

// used by pipeline timing model components:
    // modifiers
    void mem_instruction_stats(const warp_inst_t &inst);
    void decrement_atomic_count( unsigned wid, unsigned n );
    void inc_store_req( unsigned warp_id) { m_warp[warp_id].inc_store_req(); }
    void dec_inst_in_pipeline( unsigned warp_id ) { m_warp[warp_id].dec_inst_in_pipeline(); } // also used in writeback()
    void store_ack( class mem_fetch *mf );
    bool warp_waiting_at_mem_barrier( unsigned warp_id );
    void set_max_cta( const kernel_info_t &kernel );
    void warp_inst_complete(const warp_inst_t &inst);

    // accessors
    std::list<unsigned> get_regs_written( const inst_t &fvt ) const;
    const shader_core_config *get_config() const { return m_config; }
    void print_cache_stats( FILE *fp, unsigned& dl1_accesses, unsigned& dl1_misses );

    void get_cache_stats(cache_stats &cs);
    void get_L1I_sub_stats(struct cache_sub_stats &css) const;
    void get_L1D_sub_stats(struct cache_sub_stats &css) const;
    void get_L1C_sub_stats(struct cache_sub_stats &css) const;
    void get_L1T_sub_stats(struct cache_sub_stats &css) const;

    void get_icnt_power_stats(long &n_simt_to_mem, long &n_mem_to_simt) const;

// debug:
    void display_simt_state(FILE *fout, int mask ) const;
    void display_pipeline( FILE *fout, int print_mem, int mask3bit ) const;

    void incload_stat() {m_stats->m_num_loadqueued_insn[m_sid]++;}
    void incstore_stat() {m_stats->m_num_storequeued_insn[m_sid]++;}
    void incialu_stat(unsigned active_count,double latency) {
		if(m_config->gpgpu_clock_gated_lanes==false){
		  m_stats->m_num_ialu_acesses[m_sid]=m_stats->m_num_ialu_acesses[m_sid]+active_count*latency
		    + inactive_lanes_accesses_nonsfu(active_count, latency);
		}else {
        m_stats->m_num_ialu_acesses[m_sid]=m_stats->m_num_ialu_acesses[m_sid]+active_count*latency;
		}
	 }
    void inctex_stat(unsigned active_count,double latency){
    	m_stats->m_num_tex_inst[m_sid]=m_stats->m_num_tex_inst[m_sid]+active_count*latency;
    }
    void incimul_stat(unsigned active_count,double latency) {
		if(m_config->gpgpu_clock_gated_lanes==false){
		  m_stats->m_num_imul_acesses[m_sid]=m_stats->m_num_imul_acesses[m_sid]+active_count*latency
		    + inactive_lanes_accesses_nonsfu(active_count, latency);
		}else {
        m_stats->m_num_imul_acesses[m_sid]=m_stats->m_num_imul_acesses[m_sid]+active_count*latency;
		}
	 }
    void incimul24_stat(unsigned active_count,double latency) {
      if(m_config->gpgpu_clock_gated_lanes==false){
   		m_stats->m_num_imul24_acesses[m_sid]=m_stats->m_num_imul24_acesses[m_sid]+active_count*latency
		    + inactive_lanes_accesses_nonsfu(active_count, latency);
		}else {
		  m_stats->m_num_imul24_acesses[m_sid]=m_stats->m_num_imul24_acesses[m_sid]+active_count*latency;
		}
	 }
	 void incimul32_stat(unsigned active_count,double latency) {
		if(m_config->gpgpu_clock_gated_lanes==false){
		  m_stats->m_num_imul32_acesses[m_sid]=m_stats->m_num_imul32_acesses[m_sid]+active_count*latency
			 + inactive_lanes_accesses_sfu(active_count, latency);
		}else{
		  m_stats->m_num_imul32_acesses[m_sid]=m_stats->m_num_imul32_acesses[m_sid]+active_count*latency;
		}
		//printf("Int_Mul -- Active_count: %d\n",active_count);
	 }
	 void incidiv_stat(unsigned active_count,double latency) {
		if(m_config->gpgpu_clock_gated_lanes==false){
		  m_stats->m_num_idiv_acesses[m_sid]=m_stats->m_num_idiv_acesses[m_sid]+active_count*latency
			 + inactive_lanes_accesses_sfu(active_count, latency);
		}else {
		  m_stats->m_num_idiv_acesses[m_sid]=m_stats->m_num_idiv_acesses[m_sid]+active_count*latency;
		}
	 }
	 void incfpalu_stat(unsigned active_count,double latency) {
		if(m_config->gpgpu_clock_gated_lanes==false){
		  m_stats->m_num_fp_acesses[m_sid]=m_stats->m_num_fp_acesses[m_sid]+active_count*latency
			 + inactive_lanes_accesses_nonsfu(active_count, latency);
		}else {
        m_stats->m_num_fp_acesses[m_sid]=m_stats->m_num_fp_acesses[m_sid]+active_count*latency;
		}
	 }
	 void incfpmul_stat(unsigned active_count,double latency) {
		 		// printf("FP MUL stat increament\n");
      if(m_config->gpgpu_clock_gated_lanes==false){
		  m_stats->m_num_fpmul_acesses[m_sid]=m_stats->m_num_fpmul_acesses[m_sid]+active_count*latency
		    + inactive_lanes_accesses_nonsfu(active_count, latency);
		}else {
        m_stats->m_num_fpmul_acesses[m_sid]=m_stats->m_num_fpmul_acesses[m_sid]+active_count*latency;
		}
	 }
	 void incfpdiv_stat(unsigned active_count,double latency) {
		if(m_config->gpgpu_clock_gated_lanes==false){
		  m_stats->m_num_fpdiv_acesses[m_sid]=m_stats->m_num_fpdiv_acesses[m_sid]+active_count*latency
			+ inactive_lanes_accesses_sfu(active_count, latency);
		}else {
		  m_stats->m_num_fpdiv_acesses[m_sid]=m_stats->m_num_fpdiv_acesses[m_sid]+active_count*latency;
		}
	 }
	 void inctrans_stat(unsigned active_count,double latency) {
		if(m_config->gpgpu_clock_gated_lanes==false){
		  m_stats->m_num_trans_acesses[m_sid]=m_stats->m_num_trans_acesses[m_sid]+active_count*latency
			+ inactive_lanes_accesses_sfu(active_count, latency);
		}else{
		  m_stats->m_num_trans_acesses[m_sid]=m_stats->m_num_trans_acesses[m_sid]+active_count*latency;
		}
	 }

	 void incsfu_stat(unsigned active_count,double latency) {m_stats->m_num_sfu_acesses[m_sid]=m_stats->m_num_sfu_acesses[m_sid]+active_count*latency;}
	 void incsp_stat(unsigned active_count,double latency) {m_stats->m_num_sp_acesses[m_sid]=m_stats->m_num_sp_acesses[m_sid]+active_count*latency;}
	 void incmem_stat(unsigned active_count,double latency) {
		if(m_config->gpgpu_clock_gated_lanes==false){
		  m_stats->m_num_mem_acesses[m_sid]=m_stats->m_num_mem_acesses[m_sid]+active_count*latency
		    + inactive_lanes_accesses_nonsfu(active_count, latency);
		}else {
		  m_stats->m_num_mem_acesses[m_sid]=m_stats->m_num_mem_acesses[m_sid]+active_count*latency;
		}
	 }
	 void incexecstat(warp_inst_t *&inst);

	 void incregfile_reads(unsigned active_count) {m_stats->m_read_regfile_acesses[m_sid]=m_stats->m_read_regfile_acesses[m_sid]+active_count;}
	 void incregfile_writes(unsigned active_count)
	 {m_stats->m_write_regfile_acesses[m_sid]=m_stats->m_write_regfile_acesses[m_sid]+active_count;}

////DQ_code/////test_code
	// printf("write_count:%u\n",m_stats->m_write_regfile_acesses[m_sid]);}


	 void incnon_rf_operands(unsigned active_count){m_stats->m_non_rf_operands[m_sid]=m_stats->m_non_rf_operands[m_sid]+active_count;}

	 void incspactivelanes_stat(unsigned active_count) {m_stats->m_active_sp_lanes[m_sid]=m_stats->m_active_sp_lanes[m_sid]+active_count;}
	 void incsfuactivelanes_stat(unsigned active_count) {m_stats->m_active_sfu_lanes[m_sid]=m_stats->m_active_sfu_lanes[m_sid]+active_count;}
	 void incfuactivelanes_stat(unsigned active_count) {m_stats->m_active_fu_lanes[m_sid]=m_stats->m_active_fu_lanes[m_sid]+active_count;}
	 void incfumemactivelanes_stat(unsigned active_count) {m_stats->m_active_fu_mem_lanes[m_sid]=m_stats->m_active_fu_mem_lanes[m_sid]+active_count;}

	 void inc_simt_to_mem(unsigned n_flits){ m_stats->n_simt_to_mem[m_sid] += n_flits; }
	 bool check_if_non_released_reduction_barrier(warp_inst_t &inst);

	private:
	 unsigned inactive_lanes_accesses_sfu(unsigned active_count,double latency){
      return  ( ((32-active_count)>>1)*latency) + ( ((32-active_count)>>3)*latency) + ( ((32-active_count)>>3)*latency);
	 }
	 unsigned inactive_lanes_accesses_nonsfu(unsigned active_count,double latency){
      return  ( ((32-active_count)>>1)*latency);
	 }

    int test_res_bus(int latency);
    void init_warps(unsigned cta_id, unsigned start_thread, unsigned end_thread);
    virtual void checkExecutionStatusAndUpdate(warp_inst_t &inst, unsigned t, unsigned tid);
    address_type next_pc( int tid ) const;
    void fetch();
    void register_cta_thread_exit( unsigned cta_num );

    void decode();

    void issue();
    friend class scheduler_unit; //this is needed to use private issue warp.
    friend class TwoLevelScheduler;
    friend class LooseRoundRobbinScheduler;
    void issue_warp( register_set& warp, const warp_inst_t *pI, const active_mask_t &active_mask, unsigned warp_id );
    void func_exec_inst( warp_inst_t &inst );

     // Returns numbers of addresses in translated_addrs
    unsigned translate_local_memaddr( address_type localaddr, unsigned tid, unsigned num_shader, unsigned datasize, new_addr_type* translated_addrs );

    void read_operands();

    void execute();

    void writeback();
 //   m_operand_collector
    // used in display_pipeline():
    void dump_warp_state( FILE *fout ) const;
    void print_stage(unsigned int stage, FILE *fout) const;
    unsigned long long m_last_inst_gpu_sim_cycle;
    unsigned long long m_last_inst_gpu_tot_sim_cycle;

    // general information
    unsigned m_sid; // shader id
    unsigned m_tpc; // texture processor cluster id (aka, node id when using interconnect concentration)
    const shader_core_config *m_config;
    const memory_config *m_memory_config;
    class simt_core_cluster *m_cluster;

    // statistics
    shader_core_stats *m_stats;

    // CTA scheduling / hardware thread allocation
    unsigned m_n_active_cta; // number of Cooperative Thread Arrays (blocks) currently running on this shader.
    unsigned m_cta_status[MAX_CTA_PER_SHADER]; // CTAs status
    unsigned m_not_completed; // number of threads to be completed (==0 when all thread on this core completed)
    std::bitset<MAX_THREAD_PER_SM> m_active_threads;

    // thread contexts
    thread_ctx_t             *m_threadState;

    // interconnect interface
    mem_fetch_interface *m_icnt;
    shader_core_mem_fetch_allocator *m_mem_fetch_allocator;

    // fetch
    read_only_cache *m_L1I; // instruction cache
    int  m_last_warp_fetched;

    // decode/dispatch
    std::vector<shd_warp_t>   m_warp;   // per warp information array
    barrier_set_t             m_barriers;
    ifetch_buffer_t           m_inst_fetch_buffer;
    std::vector<register_set> m_pipeline_reg;
    Scoreboard               *m_scoreboard;
    opndcoll_rfu_t            m_operand_collector;

    //schedule
    std::vector<scheduler_unit*>  schedulers;

    // execute
    unsigned m_num_function_units;
    std::vector<pipeline_stage_name_t> m_dispatch_port;
    std::vector<pipeline_stage_name_t> m_issue_port;
    std::vector<simd_function_unit*> m_fu; // stallable pipelines should be last in this array
    ldst_unit *m_ldst_unit;
    static const unsigned MAX_ALU_LATENCY = 512;
    unsigned num_result_bus;
    std::vector< std::bitset<MAX_ALU_LATENCY>* > m_result_bus;

    // used for local address mapping with single kernel launch
    unsigned kernel_max_cta_per_shader;
    unsigned kernel_padded_threads_per_cta;
    // Used for handing out dynamic warp_ids to new warps.
    // the differnece between a warp_id and a dynamic_warp_id
    // is that the dynamic_warp_id is a running number unique to every warp
    // run on this shader, where the warp_id is the static warp slot.
    unsigned m_dynamic_warp_id;

    ////DQ_code///flag///

    int interleave;
    int inc;
    unsigned row_nu;
    unsigned sub_nu;
    unsigned bank;
    int n;
    unsigned dynamic_mapping;
    unsigned sub_group;
    int sub_config;
    unsigned search;
    int pipOp;

    unsigned bank_size;
    unsigned subbank_size;
    unsigned sub_searching;
    unsigned bank_searching;
    unsigned dynamic_sub;
    unsigned dynamic_bank;
    unsigned sub_nu_remapping;
    unsigned bank_remapping;
    unsigned register_valid[501][16];

    unsigned** valid;//[sub_config] [16];
    //unsigned flag_subbank[8192][16];//DQ_code//configuration subbank
    unsigned** sub_row_nu_tag;//[200][16];

    unsigned** count;//[200] [16];
    unsigned** sub_available;//[200][16];
    unsigned** sub_timing;//[200][16];
    unsigned** register_available;//[200][16];

    std::vector<unsigned> order_timing_bank;
    std::vector<unsigned> order_timing_sub;
    //unsigned sub_partition;
    int** flag;// [200] [16];///DQ_code////DQ_config//
    unsigned** register_mapping_rownu;//[200][16];
    unsigned** register_mapping_bank;//[200][16];
    //int subtag [8192][16];
    int**  row_flag;//[200][16];
    int**  bank_flag;//[200][16];
    unsigned** sub_bank_tag;//[200][16];
    int flagCount;
    std::vector<std::vector<unsigned> >reg_buf;
    std::vector<warp_inst_t>inst_buf;
    std::vector<std::vector<unsigned> >bank_buf;
    std::vector<warp_inst_t> free_inst;
    struct ssss{
    	std::vector<std::vector<unsigned> >reg_buf;
    	unsigned bank;
    	const warp_inst_t* inst_buf;
    }s;
};

class simt_core_cluster {
	//////////////////////DQ////////////////////////////
public:
    simt_core_cluster( class gpgpu_sim *gpu, 
                       unsigned cluster_id, 
                       const struct shader_core_config *config, 
                       const struct memory_config *mem_config,
                       shader_core_stats *stats,
                       memory_stats_t *mstats );

    void core_cycle();
    void icnt_cycle();

    void reinit();
    unsigned issue_block2core();
    void cache_flush();
    bool icnt_injection_buffer_full(unsigned size, bool write);
    void icnt_inject_request_packet(class mem_fetch *mf);

    // for perfect memory interface
    bool response_queue_full() {
        return ( m_response_fifo.size() >= m_config->n_simt_ejection_buffer_size );
    }
    void push_response_fifo(class mem_fetch *mf) {
        m_response_fifo.push_back(mf);
    }

    void get_pdom_stack_top_info( unsigned sid, unsigned tid, unsigned *pc, unsigned *rpc ) const;
    unsigned max_cta( const kernel_info_t &kernel );
    unsigned get_not_completed() const;
    void print_not_completed( FILE *fp ) const;
    unsigned get_n_active_cta() const;
    unsigned get_n_active_sms() const;
    gpgpu_sim *get_gpu() { return m_gpu; }

    void display_pipeline( unsigned sid, FILE *fout, int print_mem, int mask );
    void print_cache_stats( FILE *fp, unsigned& dl1_accesses, unsigned& dl1_misses ) const;

    void get_cache_stats(cache_stats &cs) const;
    void get_L1I_sub_stats(struct cache_sub_stats &css) const;
    void get_L1D_sub_stats(struct cache_sub_stats &css) const;
    void get_L1C_sub_stats(struct cache_sub_stats &css) const;
    void get_L1T_sub_stats(struct cache_sub_stats &css) const;

    void get_icnt_stats(long &n_simt_to_mem, long &n_mem_to_simt) const;

private:
    unsigned m_cluster_id;
    gpgpu_sim *m_gpu;
    const shader_core_config *m_config;
    shader_core_stats *m_stats;
    memory_stats_t *m_memory_stats;
    shader_core_ctx **m_core;

    unsigned m_cta_issue_next_core;
    std::list<unsigned> m_core_sim_order;
    std::list<mem_fetch*> m_response_fifo;
};

class shader_memory_interface : public mem_fetch_interface {
public:
    shader_memory_interface( shader_core_ctx *core, simt_core_cluster *cluster ) { m_core=core; m_cluster=cluster; }
    virtual bool full( unsigned size, bool write ) const 
    {
        return m_cluster->icnt_injection_buffer_full(size,write);
    }
    virtual void push(mem_fetch *mf) 
    {
    	m_core->inc_simt_to_mem(mf->get_num_flits(true));
        m_cluster->icnt_inject_request_packet(mf);        
    }
private:
    shader_core_ctx *m_core;
    simt_core_cluster *m_cluster;
};

class perfect_memory_interface : public mem_fetch_interface {
public:
    perfect_memory_interface( shader_core_ctx *core, simt_core_cluster *cluster ) { m_core=core; m_cluster=cluster; }
    virtual bool full( unsigned size, bool write) const
    {
        return m_cluster->response_queue_full();
    }
    virtual void push(mem_fetch *mf)
    {
        if ( mf && mf->isatomic() )
            mf->do_atomic(); // execute atomic inside the "memory subsystem"
        m_core->inc_simt_to_mem(mf->get_num_flits(true));
        m_cluster->push_response_fifo(mf);        
    }
private:
    shader_core_ctx *m_core;
    simt_core_cluster *m_cluster;
};


inline int scheduler_unit::get_sid() const { return m_shader->get_sid(); }

#endif /* SHADER_H */
