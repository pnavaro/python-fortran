program hello
use omp_lib
implicit none

integer :: n
real(8) :: x
integer, allocatable :: seed(:)
integer ::  nthreads, tid

!$OMP PARALLEL PRIVATE(NTHREADS, TID)
tid = omp_get_thread_num() ! Obtain thread number
!print *, 'Hello World from (rank,thread) = ', rank, tid

! Only master thread does this
nthreads = omp_get_num_threads()
print *, "Thread no : ", tid, " of ", nthreads

! All threads join master thread and disband
!$OMP END PARALLEL

end 
