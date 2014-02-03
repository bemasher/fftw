package fftw

// #cgo CFLAGS: -I.
// #cgo LDFLAGS: -L. -lfftw3-3
// #include <complex.h>
// #include "fftw3.h"
import "C"

import (
	"fmt"
	"reflect"
	"unsafe"
)

type plan struct {
	p C.fftw_plan
}

func (p plan) Close() {
	C.fftw_destroy_plan(p.p)
}

func (p plan) Execute() {
	C.fftw_execute(p.p)
}

type DFT1DPlan struct {
	plan
	Direction Direction
	Locality  Locality
	PlanFlags PlanFlag
	In, Out   []complex128
}

func NewDFT1D(n uint, dir Direction, locality Locality, planFlags PlanFlag) (plan DFT1DPlan) {
	plan.Direction = dir
	plan.Locality = locality
	plan.PlanFlags = planFlags

	plan.In = make([]complex128, n)
	switch locality {
	case InPlace:
		plan.Out = plan.In
	case OutOfPlace:
		plan.Out = make([]complex128, n)
	default:
		panic(fmt.Sprintf("invalid locality: %d", locality))
	}

	plan.plan.p = C.fftw_plan_dft_1d(
		C.int(len(plan.In)),
		(*C.fftw_complex)(&plan.In[0]),
		(*C.fftw_complex)(&plan.Out[0]),
		C.int(dir),
		C.uint(planFlags),
	)
	return
}

type DFTR2CPlan struct {
	plan
	Locality  Locality
	PlanFlags PlanFlag
	In        []float64
	Out       []complex128
}

func (r2c DFTR2CPlan) String() string {
	return fmt.Sprintf("{Kind:R2C Locality:%s PlanFlags:%s In:[%d]complex128 Out:[%d]float64}", r2c.Locality, r2c.PlanFlags, len(r2c.In), len(r2c.Out))
}

func NewDFTR2C(n uint, locality Locality, planFlags PlanFlag) (plan DFTR2CPlan) {
	plan.Locality = locality
	plan.PlanFlags = planFlags

	outLen := int(n>>1 + 1)

	switch locality {
	case InPlace:
		plan.In = make([]float64, n+2)
		header := *(*reflect.SliceHeader)(unsafe.Pointer(&plan.Out))
		header.Len = outLen
		header.Cap = outLen
		header.Data = uintptr(unsafe.Pointer(&plan.In[0]))
		plan.Out = *(*[]complex128)(unsafe.Pointer(&header))
	case OutOfPlace:
		plan.In = make([]float64, n)
		plan.Out = make([]complex128, outLen)
	default:
		panic(fmt.Sprintf("invalid locality: %d\n", locality))
	}

	plan.plan.p = C.fftw_plan_dft_r2c_1d(
		C.int(n),
		(*C.double)(&plan.In[0]),
		(*C.fftw_complex)(&plan.Out[0]),
		C.uint(planFlags),
	)

	return
}

type DFTC2RPlan struct {
	plan
	Locality  Locality
	PlanFlags PlanFlag
	In        []complex128
	Out       []float64
}

func (c2r DFTC2RPlan) String() string {
	return fmt.Sprintf("{Kind:C2R Locality:%s PlanFlags:%s In:[%d]complex128 Out:[%d]float64}", c2r.Locality, c2r.PlanFlags, len(c2r.In), len(c2r.Out))
}

func NewDFTC2R(n uint, locality Locality, planFlags PlanFlag) (plan DFTC2RPlan) {
	plan.Locality = locality
	plan.PlanFlags = planFlags

	plan.In = make([]complex128, n>>1+1)

	switch {
	case locality == InPlace:
		header := *(*reflect.SliceHeader)(unsafe.Pointer(&plan.Out))
		header.Len = int(n)
		header.Cap = int(n)
		header.Data = uintptr(unsafe.Pointer(&plan.In[0]))
		plan.Out = *(*[]float64)(unsafe.Pointer(&header))
	case locality == OutOfPlace:
		plan.In = make([]complex128, n>>1+1)
		plan.Out = make([]float64, n)
	default:
		panic(fmt.Sprintf("invalid locality: %d\n", locality))
	}

	plan.plan.p = C.fftw_plan_dft_c2r_1d(
		C.int(n),
		(*C.fftw_complex)(&plan.In[0]),
		(*C.double)(&plan.Out[0]),
		C.uint(planFlags),
	)

	return
}
