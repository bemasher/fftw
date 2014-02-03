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
	case InPlace:
		plan.Out = plan.In
	case OutOfPlace:
		plan.Out = make([]complex128, n)
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

// 1D Real-to-Complex DFT for transforming real data to complex data.
type DFTR2CPlan struct {
	plan
	Locality  Locality
	PlanFlags PlanFlag
	In        []float64
	Out       []complex128
}

// Plans 1D Real-to-Complex DFT of size n. For in-place transforms input and
// output arrays are backed by the same array using reflection. This is
// inherently unsafe as it requires casting an array of one type to another
// which is not allowed in the type system.
func NewDFTR2C(n uint, locality Locality, planFlags PlanFlag) (plan DFTR2CPlan) {
	plan.Locality = locality
	plan.PlanFlags = planFlags

	plan.In = make([]float64, n)

	// Allocate the input and output arrays according to given locality. In-
	// place transforms require reflection to point the input and output
	// arrays to the same data since they are of different types.
	switch locality {
	case InPlace:
		plan.In = make([]float64, n+2)
		header := *(*reflect.SliceHeader)(unsafe.Pointer(&plan.Out))
		header.Len = int(n>>1 + 1)
		header.Cap = header.Len
		header.Data = uintptr(unsafe.Pointer(&plan.In[0]))
		plan.Out = *(*[]complex128)(unsafe.Pointer(&header))
	case OutOfPlace:
		plan.Out = make([]complex128, n>>1+1)
	default:
		panic(fmt.Sprintf("invalid locality: %d\n", locality))
	}

	// Create the plan using the given arrays and flags
	plan.plan.p = C.fftw_plan_dft_r2c_1d(
		C.int(n),
		(*C.double)(&plan.In[0]),
		(*C.fftw_complex)(&plan.Out[0]),
		C.uint(planFlags),
	)

	return
}

func (r2c DFTR2CPlan) String() string {
	return fmt.Sprintf("{Kind:R2C Locality:%s PlanFlags:%s In:[%d]complex128 Out:[%d]float64}", r2c.Locality, r2c.PlanFlags, len(r2c.In), len(r2c.Out))
}

// 1D Complex-to-Real DFT for transforming complex data to real data.
type DFTC2RPlan struct {
	plan
	Locality  Locality
	PlanFlags PlanFlag
	In        []complex128
	Out       []float64
}

// Plans 1D Complex-to-Real DFT of size n. For in-place transforms input and
// output arrays are backed by the same array using reflection. This is
// inherently unsafe as it requires casting an array of one type to another
// which is not allowed in the type system.
func NewDFTC2R(n uint, locality Locality, planFlags PlanFlag) (plan DFTC2RPlan) {
	plan.Locality = locality
	plan.PlanFlags = planFlags

	plan.In = make([]complex128, n>>1+1)

	// Allocate the input and output arrays according to given locality. In-
	// place transforms require reflection to point the input and output
	// arrays to the same data since they are of different types.
	switch {
	case locality == InPlace:
		header := *(*reflect.SliceHeader)(unsafe.Pointer(&plan.Out))
		header.Len = int(n)
		header.Cap = header.Len
		header.Data = uintptr(unsafe.Pointer(&plan.In[0]))
		plan.Out = *(*[]float64)(unsafe.Pointer(&header))
	case locality == OutOfPlace:
		plan.In = make([]complex128, n>>1+1)
		plan.Out = make([]float64, n)
	default:
		panic(fmt.Sprintf("invalid locality: %d\n", locality))
	}

	// Create the plan using the given arrays and flags.
	plan.plan.p = C.fftw_plan_dft_c2r_1d(
		C.int(n),
		(*C.fftw_complex)(&plan.In[0]),
		(*C.double)(&plan.Out[0]),
		C.uint(planFlags),
	)

	return
}

func (c2r DFTC2RPlan) String() string {
	return fmt.Sprintf("{Kind:C2R Locality:%s PlanFlags:%s In:[%d]complex128 Out:[%d]float64}", c2r.Locality, c2r.PlanFlags, len(c2r.In), len(c2r.Out))
}
