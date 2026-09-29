(define (problem manipulation-task)
  (:domain manipulation)
  (:objects
    blue_can - obj
    yellow_cube - obj
    bin - location
    pot - location
  )
  (:init
    (on-table blue_can)
    (on-table yellow_cube)
  )
  (:goal (and
    (on blue_can bin)
    (on yellow_cube pot)
  ))
)
