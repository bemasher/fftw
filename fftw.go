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

// The user is required to close created plans, this points to data
// allocated in C which Go GC is unaware of.
func (p plan) Close() {
	C.fftw_destroy_plan(p.p)
}

// Execute the planned transform using data contained in the arrays allocated
// during planning.
func (p plan) Execute() {
	C.fftw_execute(p.p)
}

// 1D DFT for transforming complex data to complex data.
type DFT1DPlan struct {
	plan
	Direction Direction
	Locality  Locality
	PlanFlags PlanFlag
	In, Out   []complex128
}

// Plans 1D DFT of size n.
func NewDFT1D(n uint, dir Direction, locality Locality, planFlags PlanFlag) (plan DFT1DPlan) {
	// Store configuration for string formatting later
	plan.Direction = dir
	plan.Locality = locality
	plan.PlanFlags = planFlags

	// Allocate the input and output arrays according to given locality.
	plan.In = make([]complex128, n)
	switch locality {
	case OutOfPlace:
		plan.Out = make([]complex128, n)
	case InPlace:
		plan.Out = plan.In
	default:
		panic(fmt.Sprintf("invalid locality: %d", locality))
	}

	// Create the plan using the given arrays and flags.
	plan.plan.p = C.fftw_plan_dft_1d(
		C.int(len(plan.In)),
		(*C.fftw_complex)(&plan.In[0]),
		(*C.fftw_complex)(&plan.Out[0]),
		C.int(dir),
		C.uint(planFlags),
	)
	return
}

func (dft DFT1DPlan) String() string {
	return fmt.Sprintf("{Locality:%s PlanFlags:%s In:[%d]complex128 Out:[%d]complex128}", dft.Locality, dft.PlanFlags, len(dft.In), len(dft.Out))
}

// 1D Half-Complex DFT for transforming real data to complex data and vice versa.
type HCDFT1DPlan struct {
	plan
	Direction Direction
	Locality  Locality
	PlanFlags PlanFlag
	Real      []float64
	Complex   []complex128
}

// Plans 1D Half-Complex DFT of size n. For in-place transforms input and
// output arrays are backed by the same array using reflection. This is
// inherently unsafe because it requires casting an array of one type to another
// which is not allowed in the type system.
func NewHCDFT1D(n uint, r []float64, c []complex128, dir Direction, locality Locality, planFlags PlanFlag) (plan HCDFT1DPlan) {
	plan.Direction = dir
	plan.Locality = locality
	plan.PlanFlags = planFlags

	cmplxLen := int(n>>1 + 1)

	// Allocate the input and output arrays according to given locality. In-
	// place transforms require reflection to point the input and output
	// arrays to the same data since they are of different types.
	switch locality {
	case OutOfPlace:
		plan.Real = make([]float64, n)
		plan.Complex = make([]complex128, cmplxLen)
	case InPlace:
		plan.Real = make([]float64, n, n+2)

		header := *(*reflect.SliceHeader)(unsafe.Pointer(&plan.Complex))
		header.Len = cmplxLen
		header.Cap = cmplxLen
		header.Data = uintptr(unsafe.Pointer(&plan.Real[0]))

		plan.Complex = *(*[]complex128)(unsafe.Pointer(&header))
	case PreAlloc:
		if len(r) != int(n) {
			panic(fmt.Sprintf("invalid real array length: n:%d r:%d", n, len(r)))
		}
		if len(c) != cmplxLen {
			panic(fmt.Sprintf("invalid complex array length: n:%d r:%d", cmplxLen, len(c)))
		}

		plan.Real = r
		plan.Complex = c
	default:
		panic(fmt.Sprintf("invalid locality: %d\n", locality))
	}

	// Create the plan using the given arrays and flags
	switch dir {
	case Forward:
		plan.plan.p = C.fftw_plan_dft_r2c_1d(
			C.int(n),
			(*C.double)(&plan.Real[0]),
			(*C.fftw_complex)(&plan.Complex[0]),
			C.uint(planFlags),
		)
	case Backward:
		plan.plan.p = C.fftw_plan_dft_c2r_1d(
			C.int(n),
			(*C.fftw_complex)(&plan.Complex[0]),
			(*C.double)(&plan.Real[0]),
			C.uint(planFlags),
		)
	default:
		panic("invalid direction:" + dir.String())
	}

	return
}

func (hcdft HCDFT1DPlan) String() string {
	return fmt.Sprintf("{Direction:%s Locality:%s PlanFlags:%s Real:[%d]float64 Complex:[%d]complex128}",
		hcdft.Direction, hcdft.Locality, hcdft.PlanFlags, len(hcdft.Real), len(hcdft.Complex),
	)
}
