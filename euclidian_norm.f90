subroutine euclidian_norm (a, b, c)
  real(8), intent(in) :: a, b
  real(8), intent(out) :: c 
  c =	sqrt (a*a+b*b) 
end subroutine euclidian_norm
