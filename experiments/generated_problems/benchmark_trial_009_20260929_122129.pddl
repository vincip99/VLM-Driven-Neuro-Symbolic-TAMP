(define (problem trial-009)
  (:domain manipulation)
  (:objects
    blue_can - obj
    green_can - obj
    pot - location
  )
  (:init
    (on-table blue_can)
    (on-table green_can)
  )
  (:goal (and
    (on blue_can pot)
    (on green_can pot)
  ))
)
