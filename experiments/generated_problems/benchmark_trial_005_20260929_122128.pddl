(define (problem trial-005)
  (:domain manipulation)
  (:objects
    purple_cube - obj
    pot - location
  )
  (:init
    (on-table purple_cube)
  )
  (:goal (and
    (on purple_cube pot)
  ))
)
