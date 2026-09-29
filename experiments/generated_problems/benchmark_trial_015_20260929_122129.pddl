(define (problem trial-015)
  (:domain manipulation)
  (:objects
    yellow_cube - obj
    purple_cube - obj
    green_can - obj
    pot - location
    sorting_bin - location
  )
  (:init
    (on-table yellow_cube)
    (on-table purple_cube)
    (on-table green_can)
  )
  (:goal (and
    (on yellow_cube pot)
    (on purple_cube sorting_bin)
    (on green_can pot)
  ))
)
