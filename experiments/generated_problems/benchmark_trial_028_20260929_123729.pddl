(define (problem manipulation-task)
  (:domain manipulation)
  (:objects
    yellow_block - obj
    red_cylinder - obj
    blue_cylinder - obj
    pot - location
    bin - location
  )
  (:init
    (on-table yellow_block)
    (on-table red_cylinder)
    (on-table blue_cylinder)
  )
  (:goal (and
    (on yellow_block pot)
    (on red_cylinder bin)
    (on blue_cylinder pot)
  ))
)
