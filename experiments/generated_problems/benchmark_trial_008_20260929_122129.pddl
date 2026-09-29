(define (problem trial-008)
  (:domain manipulation)
  (:objects
    yellow_cube - obj
    purple_cube - obj
    pot - location
    sorting_bin - location
  )
  (:init
    (on-table yellow_cube)
    (on-table purple_cube)
  )
  (:goal (and
    (on yellow_cube pot)
    (on purple_cube sorting_bin)
  ))
)
