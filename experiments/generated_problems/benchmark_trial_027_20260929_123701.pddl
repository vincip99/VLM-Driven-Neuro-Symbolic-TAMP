(define (problem manipulation-task)
  (:domain manipulation)
  (:objects
    green_can - obj
    purple_cube - obj
    sorting_bin - location
    pot - location
  )
  (:init
    (on-table green_can)
    (on-table purple_cube)
  )
  (:goal (and
    (on green_can sorting_bin)
    (on purple_cube pot)
  ))
)
